# /workspace/smoke_test.py
import torch
from model.network import create_network
from model.module import TrainingModule

class Cfg: pass

cfg = Cfg()
cfg.lr = 1e-4
cfg.save_dir = '/tmp'

# model config
cfg.model = Cfg()
cfg.model.emb_dim = 512
cfg.model.latent_dim = 64
cfg.model.center_pc = True
cfg.model.block_computing = False
cfg.model.pretrain = None
cfg.model.use_diffusion = True
cfg.model.diffusion_timesteps = 100

# training config
cfg.training = Cfg()
cfg.training.diffusion_weight = 1.0
cfg.training.kl_weight = 0.1
cfg.training.save_every_n_epoch = 10


# create network
net = create_network(cfg.model, mode='train')
net = net.cuda()
# wrap module
mod = TrainingModule(cfg, net, epoch_idx=0)
mod.to('cuda')

B, N = 2, 32
robot_pc = torch.randn(B, N, 3).cuda()
object_pc = torch.randn(B, N, 3).cuda()
target_pc = torch.randn(B, N, 3).cuda()
batch = {'robot_pc': robot_pc, 'object_pc': object_pc, 'target_pc': target_pc}

# forward & loss
out = net(robot_pc, object_pc, target_pc)
print("network outputs keys:", out.keys())
loss = mod.training_step(batch, 0)
print("loss computed:", loss.item())
