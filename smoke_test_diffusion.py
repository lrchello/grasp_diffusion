import torch
from omegaconf import OmegaConf      # ★ 用 OmegaConf 读取
from model.network import Network
import numpy as np

torch.set_printoptions(precision=3, sci_mode=False)

print("========== SMOKE TEST START ==========")

# 1. 构建伪 batch
B = 2
N = 1024

fake_batch = {
    "robot_pc": torch.randn(B, N, 3).cuda(),
    "object_pc": torch.randn(B, N, 3).cuda(),
    "target_pc": torch.randn(B, N, 3).cuda(),
}

# 2. 加载配置 （★ 关键修改：使用 OmegaConf）
print("\nLoading config...")
cfg = OmegaConf.load("configs/model.yaml")   # cfg 是完整 config
model_cfg = cfg.model                        # ★ 保留 OmegaConf 对象，不转 dict

# 3. 创建网络
print("Creating network...")
net = Network(model_cfg, mode='train').cuda()
net.eval()

print("Network created successfully.")

# 4. 前向推理
print("\nRunning forward pass...")
with torch.no_grad():
    outputs = net(fake_batch["robot_pc"], fake_batch["object_pc"], fake_batch["target_pc"])

print("Forward pass done.")
print("\nNetwork output keys:", outputs.keys())

# 必须包含的 keys
required_keys = ["z", "cond", "target_pc_pred"]
missing = [k for k in required_keys if k not in outputs]

if missing:
    print("\n❌ ERROR: Missing keys:", missing)
else:
    print("\n✅ All required keys present.")

# 5. 检查 decoder 输出
pred = outputs["target_pc_pred"]
print("\nChecking decoder output shape...")
print("target_pc_pred:", pred.shape)

if pred.shape != (B, N, 3):
    print("❌ ERROR: decoder output shape incorrect!")
else:
    print("✅ decoder output shape OK.")

# 6. 检查 z latent
print("\nChecking diffusion latent z...")
z = outputs["z"]
print("z shape:", z.shape)

if z.shape[-1] != model_cfg.latent_dim:
    print("❌ ERROR: latent_dim mismatch!")
else:
    print("✅ latent_dim OK.")

# 7. 检查 DRO
print("\nChecking for dro outputs...")
dro_keys = [k for k in outputs.keys() if "dro" in k.lower()]

if dro_keys:
    print("DRO keys:", dro_keys)
    print("✅ DRO kernel working.")
else:
    print("⚠ No DRO-related keys found.")

# 8. diffusion sample（如果有）
if model_cfg.use_diffusion:
    print("\n========== DIFFUSION SAMPLING TEST ==========")
    if "z_sampled" in outputs:
        z_sampled = outputs["z_sampled"]

        print("z_sampled:", z_sampled.shape)
        pred_sampled = net.decoder(z_sampled, outputs["cond"])
        print("Sampled pc shape:", pred_sampled.shape)

        print("✅ Diffusion sampling works.")
    else:
        print("⚠ z_sampled not returned (OK if not implemented yet).")

print("\n========== SMOKE TEST COMPLETE ==========")
