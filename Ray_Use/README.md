# Ray 使用示例

这是一个 Ray 分布式计算框架的使用示例集合。每个文件都是独立可运行的。

## 环境准备

```bash
# 安装 Ray
pip install ray

# 基本示例只需要 ray，不需要额外依赖

# 如需运行特定示例，可能需要额外安装：
# PyTorch 示例: pip install torch
# Tune 示例: pip install ray[tune]
# LangChain 示例: pip install langchain langchain-openai
```

## 示例文件索引

### 入门基础

| 文件 | 内容 | 难度 |
|------|------|------|
| `01_basic_example.py` | 基础 Remote 函数、ray.get/put、资源指定 | ⭐ |
| `02_actor_example.py` | Actor（有状态对象）、Actor 方法调用 | ⭐⭐ |
| `03_parallel_compute.py` | 并行计算、任务依赖、ray.wait | ⭐⭐ |

### 进阶应用

| 文件 | 内容 | 难度 |
|------|------|------|
| `04_resource_scheduling.py` | 精细资源控制、GPU 调度 | ⭐⭐⭐ |
| `05_ray_pytorch.py` | Ray + PyTorch 分布式训练 | ⭐⭐⭐ |
| `06_ray_tune.py` | Ray Tune 超参数搜索 | ⭐⭐⭐ |
| `07_ray_langchain.py` | Ray + LangChain 并行 LLM | ⭐⭐⭐ |

## 运行方式

```bash
# 进入 Ray_Use 目录
cd Ray_Use

# 运行单个示例
python 01_basic_example.py
python 02_actor_example.py
# ...

# 或直接运行所有示例（不推荐，可能有资源冲突）
# python 01_basic_example.py & python 02_actor_example.py
```

## 快速概念回顾

### Task（远程函数）
```python
@ray.remote
def add(a, b):
    return a + b

result = add.remote(1, 2)  # 异步执行，返回 ObjectRef
print(ray.get(result))     # 获取实际结果
```

### Actor（有状态对象）
```python
@ray.remote
class Counter:
    def __init__(self):
        self.count = 0
    def increment(self):
        self.count += 1
        return self.count

counter = Counter.remote()  # 创建 Actor 实例
counter.increment.remote()  # 调用 Actor 方法
```

### 并行执行
```python
# 并行执行多个任务
futures = [some_task.remote(i) for i in range(100)]
results = ray.get(futures)  # 一次性获取所有结果
```

### 资源指定
```python
@ray.remote(num_gpus=1)  # 需要 1 个 GPU
def train_model():
    ...

@ray.remote(num_cpus=4)  # 需要 4 个 CPU
def process_data():
    ...
```

## 常见问题

### Q: 为什么任务没有真正并行执行？
A: 可能的原因：
1. 没有足够资源（CPU/GPU）
2. 任务执行太快，看不出并行效果
3. 任务之间有依赖关系

### Q: ray.init() 可以调用多次吗？
A: 可以，但后续调用会返回已初始化的集群，不会重新创建。

### Q: 如何停止 Ray 集群？
```python
ray.shutdown()
```

或在命令行：
```bash
ray stop
```

## 学习路径建议

1. **第一天**: 运行 `01_basic_example.py`，理解 `@ray.remote` 和 `ray.get()`
2. **第二天**: 运行 `02_actor_example.py`，理解 Actor 的有状态特性
3. **第三天**: 运行 `03_parallel_compute.py`，掌握并行计算的精髓
4. **之后**: 根据兴趣选择 `05_ray_pytorch.py`、`06_ray_tune.py` 或 `07_ray_langchain.py`

## 更多资源

- [Ray 官方文档](https://docs.ray.io/)
- [Ray API 参考](https://docs.ray.io/en/latest/api.html)
- [Ray GitHub](https://github.com/ray-project/ray)
