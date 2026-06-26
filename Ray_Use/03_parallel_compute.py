"""
Ray 并行计算示例 - 展示 Ray 的分布式计算能力

演示内容：
1. 并行执行多个独立任务
2. ray.wait() 等待部分任务完成
3. ray.get() 获取部分结果
4. 任务依赖关系的表达
"""

import ray
import time

ray.init()

print("=" * 60)
print("Ray 并行计算示例")
print("=" * 60)

# ============================================================
# 示例 1：批量并行执行
# ============================================================
@ray.remote
def square(x):
    """计算平方"""
    return x * x

# 传统循环方式（串行）
start = time.time()
results_serial = [square.remote(i) for i in range(10)]  # 注意：这里的 list comprehension 会立即提交所有任务
results = ray.get(results_serial)
print(f"并行执行 10 个 square 任务: {results}")
print(f"耗时: {time.time() - start:.3f}秒")

# ============================================================
# 示例 2：ray.wait() - 等待部分任务完成
# ============================================================
@ray.remote
def slow_square(x):
    """模拟耗时的平方计算"""
    time.sleep(0.5)
    return x * x

# 提交 5 个任务，每个需要 0.5 秒
tasks = [slow_square.remote(i) for i in range(5)]

# 等待前 3 个完成就返回
completed, remaining = ray.wait(tasks, num_returns=3)
print(f"\n已完成: {len(completed)} 个")
print(f"剩余: {len(remaining)} 个")

# 即使后面的任务还没完成，也可以先处理已完成的结果
completed_values = ray.get(completed)
print(f"已完成的结果: {completed_values}")

# ============================================================
# 示例 3：任务依赖 - 一个任务的输入依赖另一个任务的输出
# ============================================================
@ray.remote
def double(x):
    """翻倍"""
    return x * 2

@ray.remote
def add(a, b):
    """相加"""
    return a + b

# 方式 1：显式使用 ray.get() 获取中间结果（会阻塞）
double_result = double.remote(5)
add_result = add.remote(double_result, 3)  # 这里会等待 double_result 完成
print(f"\n方式 1 - 链式依赖: add(double(5), 3) = {ray.get(add_result)}")

# 方式 2：用 ray.get() 阻塞获取中间值
double_ref = double.remote(10)
double_value = ray.get(double_ref)  # 显式等待
final = add.remote(double_value, 20)
print(f"方式 2 - 显式依赖: add(double(10), 20) = {ray.get(final)}")

# ============================================================
# 示例 4：复杂依赖图
# ============================================================
@ray.remote
def multiply(x, y):
    return x * y

@ray.remote
def sum_three(a, b, c):
    return a + b + c

# 构建一个简单的计算图
#     multiply(2, 3)
#          |
#     multiply(?, 4)  <- 依赖上面结果
#          |
#     sum_three(?, 10, 20)  <- 也依赖上面结果
#          |
#          v

a = multiply.remote(2, 3)
b = multiply.remote(a, 4)
c = sum_three.remote(b, 10, 20)

print(f"\n计算图结果: multiply(2,3)=6 -> multiply(6,4)=24 -> sum_three(24,10,20)={ray.get(c)}")

# ============================================================
# 示例 5：并行 + 依赖混用
# ============================================================
@ray.remote
def process_batch(batch_id):
    """模拟处理一批数据"""
    time.sleep(0.3)
    return f"batch_{batch_id}_done"

@ray.remote
def aggregate(results):
    """聚合所有批次的结果"""
    return f"聚合了 {len(results)} 个批次"

# 并行处理多批数据
batch_tasks = [process_batch.remote(i) for i in range(4)]

# 等待所有批次完成，然后聚合
completed, _ = ray.wait(batch_tasks, num_returns=4)
batch_results = ray.get(completed)

# 所有批次完成后，进行聚合
final_result = aggregate.remote(batch_results)
print(f"\n并行+聚合: {ray.get(final_result)}")

# ============================================================
# 示例 6：并行归约操作
# ============================================================
@ray.remote
def sum_array(arr):
    """求和"""
    return sum(arr)

@ray.remote
def merge_sum(left, right):
    """合并两个和"""
    return left + right

# 把大数组分成小数组并行求和
large_array = list(range(100))
chunk_size = 25
chunks = [large_array[i:i+chunk_size] for i in range(0, len(large_array), chunk_size)]

# 第一步：并行求每个 chunk 的和
chunk_sums = [sum_array.remote(chunk) for chunk in chunks]
print(f"\n每个 chunk 的和: {ray.get(chunk_sums)}")

# 第二步：树形归约
while len(chunk_sums) > 1:
    # 两两配对归约
    new_chunk_sums = []
    for i in range(0, len(chunk_sums), 2):
        if i + 1 < len(chunk_sums):
            new_chunk_sums.append(merge_sum.remote(chunk_sums[i], chunk_sums[i+1]))
        else:
            new_chunk_sums.append(chunk_sums[i])
    chunk_sums = new_chunk_sums

final_sum = ray.get(chunk_sums[0])
print(f"树形归约最终和: {final_sum}")
print(f"验证: sum(range(100)) = {sum(range(100))}")

print("\n" + "=" * 60)
print("并行计算示例完成!")
print("=" * 60)
