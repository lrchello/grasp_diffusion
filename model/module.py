# model/module.py
import os
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.se3_transform import compute_link_pose
from utils.multilateration import multilateration
from utils.func_utils import calculate_depth
from utils.pretrain_utils import dist2weight, infonce_loss, mean_order

# latent encoder (resnet style) - used in PretrainingModule
try:
    from model.latent_encoder import LatentEncoder
except Exception:
    # fallback if imports are used differently in your environment
    from latent_encoder import LatentEncoder


class TrainingModule(pl.LightningModule):

    def compute_dro_loss(self, dro, batch=None, temperature=1.0, reduction='mean'):
        # dro expected shape (B, N, N) for DRO cross-entropy
        if dro.dim() != 3:
            raise ValueError(f"Expected dro shape (B,N,N), got {tuple(dro.shape)}")

        B, N, M = dro.shape
        if N != M:
            raise ValueError(f"dro must be square, got N={N}, M={M}")

        logits = dro.view(B * N, N)
        if temperature != 1.0:
            logits = logits / float(temperature)

        targets = torch.arange(N, device=dro.device).unsqueeze(0).repeat(B, 1).view(-1)
        loss = F.cross_entropy(logits, targets, reduction='none')

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss.view(B, N).mean(dim=1)

    def chamfer_loss(self, pred, gt):
        # pred, gt: (B, K, 3) - compute symmetric chamfer via pairwise cdist
        diff = torch.cdist(pred, gt)  # (B, K, K)
        min1 = diff.min(dim=2)[0].mean()  # pred -> gt
        min2 = diff.min(dim=1)[0].mean()  # gt -> pred
        return min1 + min2

    def __init__(self, cfg, network, epoch_idx):
        super().__init__()
        self.cfg = cfg
        self.network = network
        self.epoch_idx = epoch_idx

        # default lr
        self.lr = float(getattr(cfg, "lr", 1e-4))
        os.makedirs(getattr(self.cfg, "save_dir", "output/state_dict"), exist_ok=True)

    def ddp_print(self, *args, **kwargs):
        if getattr(self, "global_rank", 0) == 0:
            print(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        if 'robot_pc_initial' in batch and 'robot_pc_target' in batch and 'object_pc' in batch:
            robot_pc = batch['robot_pc_initial'].to(self.device)
            target_pc = batch['robot_pc_target'].to(self.device)
            object_pc = batch['object_pc'].to(self.device)
        elif 'robot_pc' in batch and 'target_pc' in batch and 'object_pc' in batch:
            robot_pc = batch['robot_pc'].to(self.device)
            target_pc = batch['target_pc'].to(self.device)
            object_pc = batch['object_pc'].to(self.device)
        else:
            raise KeyError(
                "training_step expects batch to contain either "
                "['robot_pc_initial','object_pc','robot_pc_target'] or "
                "['robot_pc','object_pc','target_pc']. "
                f"Found keys: {list(batch.keys())}"
            )

        outputs = self.network(robot_pc, object_pc, target_pc)

        dro = outputs.get('dro', None)
        if dro is None:
            raise RuntimeError("Network forward must return 'dro' for DRO loss computation.")

        target_pc_pred = outputs.get('target_pc_pred', None)

        mu = outputs.get('mu', None)
        logvar = outputs.get('logvar', None)
        z = outputs.get('z', None)

        cond = outputs.get('cond', None)

        try:
            dro_loss = self.compute_dro_loss(dro)
        except TypeError:
            dro_loss = self.compute_dro_loss(dro, batch)
        loss = dro_loss


        if target_pc_pred is not None:
            try:
                rec_loss = self.chamfer_loss(target_pc_pred, target_pc)
            except Exception:
                rec_loss = F.mse_loss(target_pc_pred, target_pc)
            rec_weight = float(getattr(self.cfg, "rec_weight", 1.0))
            loss = loss + rec_weight * rec_loss
            self.log('train/rec_loss', rec_loss, on_step=True, on_epoch=True, prog_bar=False)

        #3) Diffusion loss
        diffusion_loss = None
        use_diffusion_flag = False

        # 尽量用 cfg.model.use_diffusion 来控制
        cfg_model = getattr(self.cfg, "model", None)
        if cfg_model is not None and getattr(cfg_model, "use_diffusion", False):
            use_diffusion_flag = True

        if use_diffusion_flag and getattr(self.network, "diffusion", None) is not None:
            # 使用 mu 作为 z0（干净 latent）
            if mu is None:
                raise RuntimeError("use_diffusion=True but network outputs['mu'] is None, cannot form z0 for diffusion.")
            z0 = mu

            if cond is None:
                #如果网络没返回 cond，就用 0 向量
                self.ddp_print("Warning: cond not returned by network; using zeros for diffusion conditioning.")
                cond = torch.zeros(z0.size(0),
                                   getattr(self.network, 'emb_dim', z0.size(-1)),
                                   device=z0.device)

            diffusion_loss, _ = self.network.diffusion.p_losses(z0, cond)
            diff_w = float(getattr(self.cfg, "diffusion_weight", 1.0))
            loss = loss + diff_w * diffusion_loss
            self.log('train/diffusion_loss', diffusion_loss, on_step=True, on_epoch=True, prog_bar=False)

        #4) KL 正则（CVAE 的一小点约束
        if (mu is not None) and (logvar is not None):
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - torch.exp(logvar))
            # 默认用 diffusion_kl_weight，没有就退回 kl_weight
            kl_weight = float(getattr(self.cfg, "diffusion_kl_weight",
                                      getattr(self.cfg, "kl_weight", 0.01)))
            if kl_weight > 0.0:
                loss = loss + kl_weight * kl_loss
                self.log('train/kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=False)
                self.log('train/kl_weight', kl_weight, on_step=False, on_epoch=True, prog_bar=False)

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/dro_loss', dro_loss, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def on_after_backward(self):
        # safety: zero-out NaN grads
        for param in self.network.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                param.grad = None

    def on_train_epoch_end(self):
        self.epoch_idx += 1
        self.ddp_print(f"Training epoch: {self.epoch_idx}")
        if self.epoch_idx % int(getattr(self.cfg, "save_every_n_epoch", 5)) == 0:
            torch.save(
                self.network.state_dict(),
                os.path.join(getattr(self.cfg, "save_dir", "output/state_dict"),
                             f'epoch_{self.epoch_idx}.pth')
            )

    def configure_optimizers(self):
        # Lightning 默认会优化整个 LightningModule 的参数（包含 self.network）
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class PretrainingModule(pl.LightningModule):

    def __init__(self, cfg, encoder, diffusion_model=None):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.diffusion = diffusion_model  # optional

        # training hyperparams
        self.lr = float(getattr(self.cfg, "lr", 1e-4))
        self.temperature = float(getattr(self.cfg, "temperature", 0.1))

        # weights defaults
        self.contrast_w = float(getattr(self.cfg, "contrast_weight", 1.0))
        self.diffusion_w = float(getattr(self.cfg, "diffusion_weight", 1.0))
        self.kl_w = float(getattr(self.cfg, "diffusion_kl_weight",
                                  getattr(self.cfg, "kl_weight", 0.01)))

        # infer emb_dim
        emb_dim = None
        if hasattr(self.encoder, "emb_dim"):
            emb_dim = int(getattr(self.encoder, "emb_dim"))
        else:
            model_node = getattr(self.cfg, "model", None)
            if model_node is not None and hasattr(model_node, "emb_dim"):
                emb_dim = int(model_node.emb_dim)
            else:
                emb_dim = int(getattr(self.cfg, "emb_dim", 512))

        latent_dim = int(getattr(
            self.cfg,
            "latent_dim",
            getattr(getattr(self.cfg, "model", {}), "latent_dim", 64)
            if isinstance(getattr(self.cfg, "model", {}), dict)
            else getattr(self.cfg, "latent_dim", 64)
        ))

        # LatentEncoder 用来输出 mu/logvar
        self.latent_head = LatentEncoder(
            in_dim=emb_dim,
            dim=max(emb_dim, 4 * latent_dim),
            out_dim=latent_dim
        )

        self.epoch_idx = 0
        os.makedirs(getattr(self.cfg, "save_dir", "output/pretrain/state_dict"), exist_ok=True)

    def ddp_print(self, *args, **kwargs):
        if getattr(self, "global_rank", 0) == 0:
            print(*args, **kwargs)

    def configure_optimizers(self):
        params = list(self.encoder.parameters()) + list(self.latent_head.parameters())
        if self.diffusion is not None:
            params += list(self.diffusion.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # 预训练 batch： robot_pc_1 / robot_pc_2
        if 'robot_pc_1' in batch and 'robot_pc_2' in batch:
            robot_pc_1 = batch['robot_pc_1'].to(self.device)
            robot_pc_2 = batch['robot_pc_2'].to(self.device)
        else:
            if 'partial_pc' in batch and 'target_pc' in batch:
                robot_pc_2 = batch['partial_pc'].to(self.device)
                robot_pc_1 = batch['target_pc'].to(self.device)
            else:
                raise KeyError(
                    "PretrainingModule expects batch to contain "
                    "'robot_pc_1' and 'robot_pc_2' (or 'partial_pc'/'target_pc')."
                )

        # optional centering
        cfg_model = getattr(self.cfg, "model", None)
        if cfg_model is not None and getattr(cfg_model, "center_pc", False):
            robot_pc_1 = robot_pc_1 - robot_pc_1.mean(dim=1, keepdim=True)
            robot_pc_2 = robot_pc_2 - robot_pc_2.mean(dim=1, keepdim=True)

        # encoder forward
        phi_1 = self.encoder(robot_pc_1)  # (B, N, D)
        phi_2 = self.encoder(robot_pc_2)  # (B, N, D)

        # contrastive loss
        try:
            weights = dist2weight(robot_pc_1, func=lambda x: torch.tanh(10 * x))
        except Exception:
            weights = None
        contrast_loss, similarity = infonce_loss(phi_1, phi_2,
                                                 weights=weights,
                                                 temperature=self.temperature)
        mean_ord = mean_order(similarity)

        # pool & latent head -> mu / logvar
        g1 = phi_1.mean(dim=1)  # (B, D)
        g2 = phi_2.mean(dim=1)  # (B, D)

        mu, logvar = self.latent_head(g1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_sample = mu + eps * std

        total_loss = self.contrast_w * contrast_loss

        # diffusion 预训练（可选）：同样用 mu 当 z0，cond 用 g2
        diffusion_loss = None
        if self.diffusion is not None:
            z0 = mu
            cond = g2
            diffusion_loss, _ = self.diffusion.p_losses(z0, cond)
            total_loss = total_loss + float(
                getattr(self.cfg, "diffusion_weight", self.diffusion_w)
            ) * diffusion_loss

        # KL 正则
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl_w = float(getattr(self.cfg, "diffusion_kl_weight", self.kl_w))
        if kl_w > 0.0:
            total_loss = total_loss + kl_w * kl_loss

        # logging
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/contrast_loss', contrast_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train/mean_order', mean_ord, on_step=False, on_epoch=True, prog_bar=False)
        if diffusion_loss is not None:
            self.log('train/diffusion_loss', diffusion_loss, on_step=True, on_epoch=True, prog_bar=False)
        if kl_w > 0.0:
            self.log('train/kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log('train/kl_weight', kl_w, on_step=False, on_epoch=True, prog_bar=False)

        return total_loss

    def on_train_epoch_end(self):
        self.epoch_idx += 1
        self.ddp_print(f"Pretraining epoch: {self.epoch_idx}")
        if self.epoch_idx % int(getattr(self.cfg, "save_every_n_epoch", 5)) == 0:
            save_dir = getattr(self.cfg, "save_dir", "output/pretrain/state_dict")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.encoder.state_dict(), os.path.join(save_dir, f'encoder_epoch_{self.epoch_idx}.pth'))
            torch.save(self.latent_head.state_dict(), os.path.join(save_dir, f'latent_head_epoch_{self.epoch_idx}.pth'))
            if self.diffusion is not None:
                torch.save(self.diffusion.state_dict(), os.path.join(save_dir, f'diffusion_epoch_{self.epoch_idx}.pth'))

