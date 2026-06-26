"""
Ray Actor 示例 - 有状态的分布式对象

Actor 是 Ray 中的有状态计算单位，类似于一个持久的 Python 对象。
与 Task（无状态函数）不同，Actor 可以维护内部状态。

演示内容：
1. 定义 Actor 类
2. 创建 Actor 实例
3. 调用 Actor 方法
4. Actor 状态管理
"""

import ray

ray.init()

print("=" * 60)
print("Ray Actor 示例")
print("=" * 60)

# ============================================================
# 示例 1：最基本的 Actor
# ============================================================
@ray.remote
class Counter:
    """
    一个简单的计数器 Actor

    Actor 的特点：
    - 通过 .remote() 创建实例
    - 方法通过 .method.remote() 调用
    - 内部状态在多个调用之间保持
    """

    def __init__(self):
        # __init__ 只在创建时执行一次
        self.count = 0
        print("Counter Actor 创建!")

    def increment(self):
        """增加计数并返回当前值"""
        self.count += 1
        return self.count

    def get_count(self):
        """获取当前计数"""
        return self.count

    def reset(self):
        """重置计数"""
        self.count = 0


# 创建 Actor 实例
counter = Counter.remote()

# 调用 Actor 方法（注意：方法也要加 .remote()）
print(f"初始计数: {ray.get(counter.get_count.remote())}")

ray.get(counter.increment.remote())
ray.get(counter.increment.remote())
ray.get(counter.increment.remote())

print(f"调用 3 次 increment 后: {ray.get(counter.get_count.remote())}")

# 重置
ray.get(counter.reset.remote())
print(f"重置后: {ray.get(counter.get_count.remote())}")

# ============================================================
# 示例 2：Actor 传参
# ============================================================
@ray.remote
class Accumulator:
    """带初始值的累加器"""

    def __init__(self, initial_value: int):
        self.value = initial_value
        print(f"Accumulator 创建，初始值: {initial_value}")

    def add(self, x: int):
        self.value += x
        return self.value

    def get_value(self):
        return self.value


# 创建时传入参数
acc = Accumulator.remote(100)
print(f"累加器初始值: {ray.get(acc.get_value.remote())}")
print(f"+50 = {ray.get(acc.add.remote(50))}")
print(f"+30 = {ray.get(acc.add.remote(30))}")

# ============================================================
# 示例 3：多个 Actor 并行
# ============================================================
@ray.remote
class Poker:
    """模拟一个扑克玩家"""

    def __init__(self, name: str):
        self.name = name
        self.cards = []

    def receive_card(self, card: str):
        self.cards.append(card)
        return f"{self.name} 收到 {card}"

    def show_cards(self):
        return f"{self.name}: {self.cards}"


# 同时创建 4 个玩家 Actor
players = [Poker.remote(f"玩家{i}") for i in range(4)]

# 发牌（并发）
cards = ["红桃A", "黑桃K", "方块Q", "梅花J", "红桃10", "黑桃9", "方块8", "梅花7"]
future_results = []
for i, card in enumerate(cards):
    # 轮流给每个玩家发牌
    future_results.append(players[i % 4].receive_card.remote(card))

# 等待所有发牌完成
ray.get(future_results)

# 展示每个玩家的手牌
for player in players:
    print(ray.get(player.show_cards.remote()))

# ============================================================
# 示例 4：Actor 与外部数据交互
# ============================================================
@ray.remote
class DataProcessor:
    """数据处理器 Actor，维护处理状态"""

    def __init__(self):
        self.processed_count = 0
        self.total_sum = 0

    def process(self, data: list):
        """处理一批数据"""
        result = sum(data)
        self.processed_count += 1
        self.total_sum += result
        return result

    def get_stats(self):
        """获取统计信息"""
        return {
            "processed_batches": self.processed_count,
            "total_sum": self.total_sum
        }


processor = DataProcessor.remote()

# 处理多批数据
batches = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
results = ray.get([processor.process.remote(b) for b in batches])

print(f"每批处理结果: {results}")
print(f"统计信息: {ray.get(processor.get_stats.remote())}")

# ============================================================
# 示例 5：Actor 方法的并发控制
# ============================================================
@ray.remote
class ConcurrentCounter:
    """支持并发访问的计数器"""

    def __init__(self):
        self.count = 0

    @ray.method(max_concurrency=2)  # 最多同时有 2 个方法调用
    def increment(self):
        import time
        time.sleep(0.1)  # 模拟耗时操作
        self.count += 1
        return self.count

    @ray.method(max_concurrency=2)
    def get_count(self):
        return self.count


concurrent_counter = ConcurrentCounter.remote()

# 如果没有 max_concurrency=2，这些调用会串行执行
# 有了它，其中 2 个可以并行执行
futures = [concurrent_counter.increment.remote() for _ in range(5)]
print(f"5 次并发 increment 结果: {ray.get(futures)}")

print("\n" + "=" * 60)
print("Actor 示例完成!")
print("=" * 60)
