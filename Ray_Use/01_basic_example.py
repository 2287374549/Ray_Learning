"""
Ray 基础示例 - 最简单的入门例子

演示内容：
1. ray.init() 初始化
2. @ray.remote 装饰器创建远程函数
3. ray.get() 获取结果
4. ray.put() 存储对象
"""

import ray

# ============================================================
# 第一步：初始化 Ray
# ============================================================
# ray.init() 会启动本地 Ray 集群（单节点模式）
# 如果不调用，其他 ray 函数会自动调用它
ray.init()

print("Ray 已启动!")
print(f"head_node_id: {ray.get_runtime_context().get_node_id()}")

# ============================================================
# 示例 1：最基本的远程函数
# ============================================================
@ray.remote
def add(a, b):
    """简单的加法函数，会在单独的 worker 进程中执行"""
    return a + b

# 调用远程函数 - 注意要加 .remote()
result_future = add.remote(1, 2)
result = ray.get(result_future)  # 从分布式对象存储中获取结果
print(f"task 1: basic remote")
print(f"add.remote(1, 2) = {result}")

# ============================================================
# 示例 2：并行执行多个任务
# ============================================================
import time

@ray.remote
def slow_add(a, b):
    """模拟耗时的计算"""
    import time
    time.sleep(0.5)
    return a + b

# 顺序执行
print(f"task 2_1: 顺序")
start_time = time.time()  # 记录开始时间
result1 = slow_add.remote(0, 1)
result2 = slow_add.remote(1, 2)
result3 = slow_add.remote(2, 3)
result4 = slow_add.remote(3, 4)

print(ray.get(result1), ray.get(result2), ray.get(result3), ray.get(result4))
end_time = time.time()  # 记录结束时间
print(f"顺序执行耗时: {end_time - start_time} 秒")


# 并行执行
print(f"task 2_2: 并行")
start_time = time.time()  # 记录开始时间
futures = [slow_add.remote(i, i+1) for i in range(4)]
results = ray.get(futures)
print(f"并行执行 4 个任务: {results}")
end_time = time.time()  # 记录结束时间
print(f"并行执行耗时: {end_time - start_time} 秒")


# ============================================================
# 示例 3：ray.put() 将对象存入分布式存储
# ============================================================
data = [1, 2, 3, 4, 5]

# ray.put() 将对象存入每个节点的本地存储
data_ref = ray.put(data)
print(f"ray.put([1,2,3,4,5]) -> ObjectRef: {data_ref}")

# 这个 ObjectRef 可以传给其他 remote 函数
@ray.remote
def process_data(data):
    """接收 ObjectRef，自动从分布式存储获取数据"""
    return sum(data)

result = ray.get(process_data.remote(data_ref))
print(f"sum(data via ObjectRef) = {result}")

# ============================================================
# 示例 4：指定资源（CPU/GPU）
# ============================================================
@ray.remote(num_cpus=2)
def use_two_cpus():
    """这个任务会占用 2 个 CPU"""
    import os
    return os.cpu_count()

@ray.remote(num_gpus=1)
def use_one_gpu():
    """这个任务会占用 1 个 GPU（如果有用 GPU 的话）"""
    import torch
    return torch.cuda.is_available()

# 注意：如果没有足够的 CPU/GPU，这些任务会等待
# cpu_result = ray.get(use_two_cpus.remote())
# gpu_result = ray.get(use_one_gpu.remote())

print("\nRay 基础示例完成!")
print("运行 'ray stop' 可以停止 Ray 集群")
