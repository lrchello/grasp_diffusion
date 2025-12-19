import torch
from model.module import TrainingModule

# 假设我们有 module 实例 `mod` 已经存在；这里只做函数独立测试
B, N = 2, 8
# 构造一个 dro，使得对角线分数大
dro = torch.randn(B, N, N)
for b in range(B):
    for i in range(N):
        dro[b, i, i] += 3.0  # boost diagonal so correct class is highest

# fake batch (only for shape)
batch = {
    'robot_pc': torch.randn(B, N, 3),
    'target_pc': torch.randn(B, N, 3)
}

# run compute_dro_loss standalone (if defined as member you can call via module instance)
# Example: loss = mod.compute_dro_loss(dro, batch)
# For quick standalone test, call the function above if placed in module.py scope.
print("dro diag max check:", (dro.argmax(dim=-1) == torch.arange(N).unsqueeze(0)).all())
# If using the method bound to an object `mod`, do:
# print("loss:", mod.compute_dro_loss(dro, batch).item())

