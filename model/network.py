import os
import torch
import torch.nn as nn

from model.encoder import Encoder, CvaeEncoder
from model.transformer import Transformer
from model.mlp import MLPKernel
from model.latent_encoder import LatentEncoder
from model.diffusion import LatentDiffusion
from model.grasp_decoder import StrongDecoder  # or FoldingDecoder, PTDecoder

def create_encoder_network(emb_dim, pretrain=None, device=torch.device('cpu')) -> nn.Module:
    encoder = Encoder(emb_dim=emb_dim)
    if pretrain is not None:
        print(f"******** Load embedding network pretrain from <{pretrain}> ********")
        encoder.load_state_dict(
            torch.load(
                os.path.join(os.path.dirname(__file__), f"../ckpt/pretrain/{pretrain}"),
                map_location=device
            )
        )
    return encoder


class Network(nn.Module):
    """
    Network that integrates:
      - encoders (robot/object)
      - transformers for cross-attention
      - CVAE-style point encoder -> mu, logvar (we treat mu as z0, the 'clean' latent)
      - optional LatentDiffusion model
      - dro kernel & decoder

    Behavior:
      - mode == 'train': network requires `target_pc` (grasp GT) and will compute `mu, logvar` and z0=mu.
          * if diffusion is present, we use z = z0 for reconstruction/dro (and diffusion is trained
            outside this module via diffusion.p_losses(z0, cond, t))
          * else (no diffusion) we sample z ~ N(mu, sigma) as usual VAE.
      - mode != 'train' (eval/validate): if diffusion exists we call diffusion.sample(...) to get z,
        otherwise sample z from standard normal.
    Outputs includes keys: 'dro','mu','logvar','cond','z','z0','z_expanded','target_pc_pred'
    """
    def __init__(self, cfg, mode):
        super(Network, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.emb_dim = cfg.emb_dim
        self.latent_dim = cfg.latent_dim

        # encoders
        self.encoder_robot = create_encoder_network(emb_dim=cfg.emb_dim, pretrain=cfg.pretrain)
        self.encoder_object = create_encoder_network(emb_dim=cfg.emb_dim)

        # transformers
        self.transformer_robot = Transformer(emb_dim=cfg.emb_dim)
        self.transformer_object = Transformer(emb_dim=cfg.emb_dim)

        # CVAE encoder (point encoder used to produce latent features)
        self.point_encoder = CvaeEncoder(
            emb_dims=cfg.emb_dim,
            output_channels=2 * cfg.latent_dim,
            feat_dim=cfg.emb_dim
        )
        self.latent_encoder = LatentEncoder(in_dim=2 * cfg.latent_dim, dim=4 * cfg.latent_dim, out_dim=cfg.latent_dim)

        # kernel
        self.kernel = MLPKernel(cfg.emb_dim + cfg.latent_dim)

        # dro_decoder
        self.decoder = StrongDecoder(
            impl='pt',
            latent_dim=cfg.latent_dim,
            cond_dim=cfg.emb_dim,
            hidden_dim=256,
            K=getattr(cfg, 'K', 256),
            n_folding_layers=2,
            n_attn_layers=2,
            n_heads=4
        )

        # diffusion model (optional)
        if getattr(cfg, 'use_diffusion', False):
            cond_dim = cfg.emb_dim
            timesteps = getattr(cfg, 'diffusion_timesteps', 1000)
            # LatentDiffusion internally handles device via register_buffer for betas etc.
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.diffusion = LatentDiffusion(latent_dim=cfg.latent_dim, cond_dim=cond_dim, timesteps=timesteps, device=device)
        else:
            self.diffusion = None

    def forward(self, robot_pc, object_pc, target_pc=None):
        """
        Forward pass.
        - robot_pc: (B, N, 3)
        - object_pc: (B, M, 3)
        - target_pc: (B, K_gt, 3) required in train mode
        Returns dict with 'dro','mu','logvar','cond','z','z0','z_expanded','target_pc_pred'
        """
        # center if required
        if self.cfg.center_pc:
            robot_pc = robot_pc - robot_pc.mean(dim=1, keepdim=True)

        # encode
        robot_embedding = self.encoder_robot(robot_pc)   # (B, N, D)
        object_embedding = self.encoder_object(object_pc)  # (B, M, D)

        if self.cfg.pretrain is not None:
            # freeze encoder if pretrain specified (consistent with your previous code)
            robot_embedding = robot_embedding.detach()

        # transformer cross-attention
        transformer_robot_outputs = self.transformer_robot(robot_embedding, object_embedding)
        transformer_object_outputs = self.transformer_object(object_embedding, robot_embedding)
        robot_embedding_tf = robot_embedding + transformer_robot_outputs["src_embedding"]
        object_embedding_tf = object_embedding + transformer_object_outputs["src_embedding"]

        # unified conditioning vector (mean-pooled)
        cond = robot_embedding_tf.mean(dim=1)  # (B, emb_dim)

        # placeholders
        mu, logvar = None, None
        z = None
        z0 = None  # "clean" latent from encoder (CVAE)

        #training branch: need target_pc
        if self.mode == 'train':
            if target_pc is None:
                raise ValueError("target_pc must be provided in train mode")

            # Construct grasp_pc and grasp_emb
            grasp_pc = torch.cat([target_pc, object_pc], dim=1)  # (B, K_gt, 3)
            grasp_emb = torch.cat([robot_embedding_tf, object_embedding_tf], dim=1)  # (B, N+M, D)

            # point_encoder expects (B, N_points, feat_dim)
            latent_feat = self.point_encoder(torch.cat([grasp_pc, grasp_emb], dim=-1))
            mu, logvar = self.latent_encoder(latent_feat)  # (B, latent_dim)

            # treat mu as "clean latent" z0 (this is the target we teach diffusion to predict)
            z0 = mu

            if self.diffusion is not None:
                # during diffusion training we use z0 as the latent passed downstream (reconstruction / dro)
                # diffusion's training loss must be computed by outer training loop using z0 and cond:
                #    diffusion_loss = diffusion.p_losses(z0, cond, t)
                z = z0
            else:
                # standard CVAE sampling (reparameterization)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std

        else:
            # eval / sample mode: do not use encoder for generation if diffusion exists;
            # generate latent z using diffusion.sample(cond). If diffusion is missing, sample standard normal.
            if self.diffusion is not None:
                # diffusion.sample should accept cond and device; ensure cond is on correct device
                try:
                    z = self.diffusion.sample(batch_size=robot_pc.shape[0], cond=cond, device=cond.device)
                except Exception:
                    # fallback: if diffusion sampling fails for some reason, fall back to gaussian
                    z = torch.randn(robot_pc.shape[0], self.latent_dim, device=cond.device)
            else:
                z = torch.randn(robot_pc.shape[0], self.latent_dim, device=robot_pc.device)

        # expand z to per-point for robot/object respectively
        N = robot_embedding_tf.shape[1]
        M = object_embedding_tf.shape[1]

        z_expanded_A = z.unsqueeze(1).repeat(1, N, 1)  # (B, N, latent_dim)
        z_expanded_B = z.unsqueeze(1).repeat(1, M, 1)  # (B, M, latent_dim)

        Phi_A = torch.cat([robot_embedding_tf, z_expanded_A], dim=-1)  # (B, N, emb+latent)
        Phi_B = torch.cat([object_embedding_tf, z_expanded_B], dim=-1)  # (B, M, emb+latent)

        # Compute dro (block or full)
        if self.cfg.block_computing:
            B, N_, D = Phi_A.shape
            assert N_ % 4 == 0, "block_computing expects N divisible by 4 (or change block_num)"
            block_num = 4
            N_block = N_ // block_num

            dro = torch.zeros([B, N_, M], dtype=torch.float32, device=Phi_A.device)
            for A_i in range(block_num):
                a0 = A_i * N_block
                a1 = a0 + N_block
                Phi_A_block = Phi_A[:, a0:a1, :]  # (B, N_block, D)
                for B_i in range(block_num):
                    b0 = B_i * N_block
                    b1 = b0 + N_block
                    # Note: if M != N_, iterating over same block_num may be incorrect.
                    # Here we assume M is divisible by N_block when using block_computing.
                    Phi_B_block = Phi_B[:, b0:b1, :]  # (B, N_block, D)
                    # reshape for kernel: (B * N_block * N_block, D)
                    Phi_A_r = Phi_A_block.unsqueeze(2).repeat(1, 1, (b1 - b0), 1).reshape(B * N_block * (b1 - b0), D)
                    Phi_B_r = Phi_B_block.unsqueeze(1).repeat(1, (a1 - a0), 1, 1).reshape(B * N_block * (b1 - b0), D)
                    out_block = self.kernel(Phi_A_r, Phi_B_r).reshape(B, (a1 - a0), (b1 - b0))
                    dro[:, a0:a1, b0:b1] = out_block
        else:
            # full compute (may be memory heavy)
            B, N_, D = Phi_A.shape
            _, M_, _ = Phi_B.shape
            Phi_A_r = Phi_A.unsqueeze(2).repeat(1, 1, M_, 1).reshape(B * N_ * M_, D)
            Phi_B_r = Phi_B.unsqueeze(1).repeat(1, N_, 1, 1).reshape(B * N_ * M_, D)
            dro = self.kernel(Phi_A_r, Phi_B_r).reshape(B, N_, M_)

        # decoder: choose z_for_decode (prefer z computed above) and decode with cond
        z_for_decode = z if z is not None else (mu if mu is not None else torch.randn(cond.size(0), self.latent_dim, device=cond.device))
        target_pc_pred = self.decoder(z_for_decode, cond)  # (B, K, 3)

        outputs = {
            'dro': dro,
            'mu': mu,
            'logvar': logvar,
            'cond': cond,
            'z': z,               # latent actually used downstream (either z0 in train or sampled z in eval)
            'z0': z0,             # clean latent from encoder (mu). Important: used as diffusion target during training
            'z_expanded': z_expanded_A,
            'target_pc_pred': target_pc_pred
        }

        return outputs


def create_network(cfg, mode):
    return Network(cfg=cfg, mode=mode)
