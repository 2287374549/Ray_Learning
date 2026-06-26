"""
Ray + PyTorch 分布式训练示例

演示内容：
1. 使用 Ray 管理分布式 GPU 资源
2. 并行执行 PyTorch 训练任务
3. 数据并行处理的模式
"""

import ray
import torch
import torch.nn as nn
import time

ray.init()

print("=" * 60)
print("Ray + PyTorch 分布式训练示例")
print("=" * 60)

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"\nCUDA 可用: {cuda_available}")
if cuda_available:
    print(f"GPU 数量: {torch.cuda.device_count()}")

# ============================================================
# 示例 1：简单的分布式矩阵乘法
# ============================================================
print("\n--- 示例 1：分布式矩阵运算 ---")

@ray.remote
def matrix_multiply_remote(A, B):
    """
    在远程 worker 上执行矩阵乘法
    使用 CPU 计算（避免 GPU 资源竞争）
    """
    A_tensor = torch.tensor(A)
    B_tensor = torch.tensor(B)
    result = torch.mm(A_tensor, B_tensor)
    return result.numpy().tolist()

# 创建两个矩阵
A = [[1, 2], [3, 4], [5, 6]]
B = [[7, 8, 9], [10, 11, 12]]

# 并行执行 3 个矩阵乘法（每行乘以 B）
row_futures = []
for i in range(len(A)):
    row = [A[i]]  # 取第 i 行作为单独矩阵
    row_futures.append(matrix_multiply_remote.remote(row, B))

# 获取所有结果
results = ray.get(row_futures)
print(f"矩阵 A ({len(A)}x{len(A[0])}) @ B ({len(B)}x{len(B[0])}) = ")
for r in results:
    print(f"  {r}")

# ============================================================
# 示例 2：数据并行处理（模拟分布式训练）
# ============================================================
print("\n--- 示例 2：数据并行处理 ---")

# 模拟的数据批次
def generateFakeBatch(batch_id, batch_size=8):
    """生成假数据批次"""
    return torch.randn(batch_size, 10), torch.randint(0, 2, (batch_size,))

@ray.remote
def train_on_batch(batch_id, model_state_dict=None):
    """
    在单个批次上训练（模拟）
    返回假的 loss 值作为演示
    """
    # 重建模型
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    if model_state_dict:
        model.load_state_dict(model_state_dict)

    # 生成假数据
    inputs, targets = generateFakeBatch(batch_id)

    # 模拟前向传播
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, targets)

    # 模拟反向传播（返回梯度作为结果）
    return loss.item()

# 模拟一个训练循环
num_epochs = 3
num_batches = 6

print(f"模拟训练 {num_epochs} 个 epoch，每 epoch {num_batches} 个批次")

for epoch in range(num_epochs):
    # 并行训练所有批次
    futures = [train_on_batch.remote(i) for i in range(num_batches)]
    losses = ray.get(futures)

    avg_loss = sum(losses) / len(losses)
    print(f"Epoch {epoch+1}/{num_epochs}: 平均 Loss = {avg_loss:.4f}")

# ============================================================
# 示例 3：参数同步（简化版的分布式训练）
# ============================================================
print("\n--- 示例 3：参数同步模式 ---")

@ray.remote
class ParameterServer:
    """
    参数服务器 - 维护全局模型参数
    类似分布式训练中的 Parameter Server 架构
    """

    def __init__(self):
        # 初始化模型参数
        self.model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.train_step = 0

    def get_weights(self):
        """返回当前权重"""
        return self.model.state_dict()

    def apply_gradients(self, gradients):
        """
        应用梯度更新参数
        这是一个简化版本，实际中需要对梯度进行处理
        """
        # 重建优化器（每次都重建是为了简化）
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        # 设置梯度
        for grad, param in zip(gradients, self.model.parameters()):
            param.grad = grad

        # 更新参数
        optimizer.step()
        self.train_step += 1

        return self.train_step

# 创建参数服务器
ps = ParameterServer.remote()
initial_weights = ray.get(ps.get_weights.remote())
print(f"参数服务器已创建，当前训练步数: {ray.get(ps.get_train_step.remote())}")

@ray.remote
def compute_gradients(batch_id, weights):
    """
    计算梯度（模拟）
    实际中这会在 GPU 上运行
    """
    # 重建模型并加载权重
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    model.load_state_dict(weights)

    # 生成假数据
    inputs = torch.randn(8, 10)
    targets = torch.randint(0, 2, (8,))

    # 前向传播
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, targets)

    # 反向传播
    loss.backward()

    # 提取梯度
    gradients = [param.grad.clone() for param in model.parameters()]
    return gradients

# 模拟几轮训练
print("\n模拟参数服务器训练:")
for step in range(3):
    # 1. 获取当前权重
    weights = ray.get(ps.get_weights.remote())

    # 2. 并行计算多个 worker 的梯度
    grad_futures = [compute_gradients.remote(i, weights) for i in range(2)]
    all_gradients = ray.get(grad_futures)

    # 3. 平均梯度
    avg_gradients = []
    for grads in zip(*all_gradients):
        avg_gradients.append(sum(grads) / len(grads))

    # 4. 更新参数
    train_step = ray.get(ps.apply_gradients.remote(avg_gradients))
    print(f"  Step {step+1}: 训练步数 = {train_step}")

# ============================================================
# 示例 4：模拟 PyTorch DDP 风格的训练
# ============================================================
print("\n--- 示例 4：模拟 DDP 多卡训练 ---")

@ray.remote(num_cpus=1)  # 每个 trainer 占用 1 个 CPU（生产环境用 GPU）
class DistributedTrainer:
    """
    模拟分布式数据并行训练器
    类似 PyTorch DDP (DistributedDataParallel)
    """

    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

        # 初始化本地模型
        self.model = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def train_step(self, batch_id):
        """执行一个训练步骤"""
        inputs = torch.randn(4, 10)
        targets = torch.randint(0, 2, (4,))

        logits = self.model(inputs)
        loss = torch.nn.functional.cross_entropy(logits, targets)

        return {
            "rank": self.rank,
            "loss": loss.item(),
            "batch_id": batch_id
        }

    def get_model_weights(self):
        """获取本地模型权重（用于同步）"""
        return {k: v.clone() for k, v in self.model.state_dict().items()}


# 模拟 4 卡训练
world_size = 4
trainers = [DistributedTrainer.remote(rank=i, world_size=world_size) for i in range(world_size)]

# 训练几个步骤
for step in range(2):
    # 并行在所有 trainer 上执行训练
    futures = [trainer.train_step.remote(step) for trainer in trainers]
    results = ray.get(futures)

    # 展示结果
    losses = [r["loss"] for r in results]
    avg_loss = sum(losses) / len(losses)
    print(f"  Step {step}: Rank 0 loss={results[0]['loss']:.4f}, 平均={avg_loss:.4f}")

print("\n" + "=" * 60)
print("Ray + PyTorch 示例完成!")
print("=" * 60)
