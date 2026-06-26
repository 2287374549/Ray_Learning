"""
Ray 资源调度示例 - 控制任务执行的资源分配

演示内容：
1. 为任务指定 CPU/GPU 资源
2. 自定义资源配置
3. 资源约束下的任务调度
4. 多 GPU 并行训练模拟
"""

import ray

ray.init()

print("=" * 60)
print("Ray 资源调度示例")
print("=" * 60)

# 查看可用资源
print(f"\n集群总资源: {ray.available_resources()}")
print(f"集群总资源（所有）: {ray.cluster_resources()}")

# ============================================================
# 示例 1：基础资源指定
# ============================================================
@ray.remote(num_cpus=1)
def use_one_cpu():
    """占用 1 个 CPU"""
    import os
    return f"使用 CPU 数量: 1, 可用: {ray.available_resources().get('CPU', 0)}"

@ray.remote(num_cpus=2)
def use_two_cpus():
    """占用 2 个 CPU"""
    return f"使用 CPU 数量: 2, 可用: {ray.available_resources().get('CPU', 0)}"

# 运行任务
result1 = ray.get(use_one_cpu.remote())
result2 = ray.get(use_two_cpus.remote())
print(f"\n任务 1: {result1}")
print(f"任务 2: {result2}")

# ============================================================
# 示例 2：自定义资源
# ============================================================
@ray.remote(resources={"CustomResource": 1})
def use_custom_resource():
    """使用自定义资源"""
    return "使用了 CustomResource"

@ray.remote(resources={"CustomResource": 2})
def use_two_custom_resources():
    """使用 2 个自定义资源"""
    return "使用了 2 个 CustomResource"

# 自定义资源的用途：在特定机器上调度任务
# 比如有些机器有 SSD，有些有特定软件
print(f"\n自定义资源测试:")
print(ray.get(use_custom_resource.remote()))

# ============================================================
# 示例 3：GPU 调度（如果有 GPU 的话）
# ============================================================
@ray.remote(num_gpus=1)
def gpu_task(task_id):
    """占用 1 个 GPU 的任务"""
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        return f"Task {task_id}: GPU 可用，设备数: {gpu_count}"
    return f"Task {task_id}: 无 GPU"

# 注意：这个任务只有在有 GPU 的环境才能实际运行
# 如果没有 GPU，任务会等待（或者报错，取决于配置）
try:
    # 只有在有 GPU 时才实际运行
    if ray.available_resources().get("GPU", 0) > 0:
        result = ray.get(gpu_task.remote(1))
        print(f"\nGPU 任务: {result}")
    else:
        print("\n当前环境无 GPU，跳过 GPU 任务测试")
except Exception as e:
    print(f"\nGPU 任务执行出错: {e}")

# ============================================================
# 示例 4：GPU 数量精确控制
# ============================================================
@ray.remote(num_gpus=0.5)
def half_gpu_task():
    """占用半个 GPU（用于共享 GPU）"""
    return "使用半个 GPU"

# 在某些场景下，可以让多个小任务共享同一个 GPU
# 特别是深度学习中的小模型推理场景
# print(f"\n共享 GPU 测试: {ray.get(half_gpu_task.remote())}")

# ============================================================
# 示例 5：placement_options - 控制任务放置
# ============================================================
@ray.remote
def get_node_id():
    """获取当前执行节点的 ID"""
    return ray.get_runtime_context().get_node_id()

# 获取所有节点上执行的结果
futures = [get_node_id.remote() for _ in range(4)]
node_ids = ray.get(futures)
unique_nodes = set(node_ids)
print(f"\n任务执行的节点数: {len(unique_nodes)}")
print(f"节点 ID 列表: {unique_nodes}")

# ============================================================
# 示例 6：资源 bundles - 用于 placement groups
# ============================================================
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# placement group 用于保证一组任务调度到同一组节点上
# 常用于 Actor 的稳定性保证
@ray.remote
class ResourceAwareActor:
    """知道自己在哪个节点上运行的 Actor"""

    def __init__(self):
        self.node_id = ray.get_runtime_context().get_node_id()

    def get_location(self):
        return self.node_id

# 创建 2 个 Actor（可能会在同一节点或不同节点，取决于调度）
actors = [ResourceAwareActor.remote() for _ in range(2)]
locations = ray.get([a.get_location.remote() for a in actors])
print(f"\nActor 分布的节点数: {len(set(locations))}")
print(f"Actor 所在节点: {locations}")

print("\n" + "=" * 60)
print("资源调度示例完成!")
print("=" * 60)
