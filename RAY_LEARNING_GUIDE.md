# Ray 代码架构学习指南

## 1. 项目整体结构

```
ray-master/
├── python/          # Python API 层（主要学习入口）
│   └── ray/
│       ├── __init__.py           # 公共 API 导出
│       ├── _private/             # 核心内部实现
│       │   ├── worker.py         # Worker 实现 (~3700行)
│       │   ├── services.py       # 进程管理（启动 raylet、gcs 等）
│       │   ├── node.py           # 节点管理
│       │   ├── serialization.py  # 对象序列化
│       │   └── function_manager.py  # 函数/Actor 管理
│       ├── actor.py              # Actor 类装饰器
│       ├── remote_function.py    # Remote Function 装饰器
│       ├── runtime_context.py    # 运行时上下文
│       ├── job_config.py         # Job 配置
│       └── train/ data/ tune/ serve/  # AI 库
├── src/             # C++ 核心实现
│   └── ray/
│       ├── core_worker/          # 核心 Worker（执行 Task/Actor）
│       ├── raylet/               # 本地节点调度
│       ├── gcs/                  # 全局控制服务（集群状态）
│       ├── object_manager/       # 分布式对象存储
│       ├── rpc/                  # gRPC 通信层
│       └── protobuf/             # 协议缓冲区定义
└── rllib/           # 强化学习库
```

---

## 2. 核心概念

Ray 是一个分布式执行引擎，核心概念有三个：

| 概念 | 说明 | 代码位置 |
|------|------|----------|
| **Task** | 在远程 workers 上执行的函数 | `remote_function.py` |
| **Actor** | 有状态的 worker 进程 | `actor.py` |
| **ObjectRef** | 分布式对象的引用 | `__init__.py` (从 `_raylet` 导入) |

---

## 3. 推荐学习路径

### 第一步：理解入口点

**文件：** `python/ray/__init__.py`

这是用户 `import ray` 首先加载的文件，展示了 Ray 对外暴露的 API：

```python
# 主要导入的函数（按重要性排序）：
from ray._private.worker import (
    init,       # 初始化 Ray 集群
    get,        # 获取远程对象
    put,        # 将对象存入对象存储
    remote,     # 装饰器，将函数变为远程函数
    wait,       # 等待 ObjectRef 就绪
    kill,       # 杀死 actor
    cancel,     # 取消任务
)

from ray.actor import method  # Actor 方法装饰器

# 各种 ID 类型
from ray._raylet import (
    ActorID, ObjectRef, NodeID, JobID, WorkerID, ...
)
```

**关键发现：**
- `ray.init()` 是启动 Ray 的入口
- `ray.get()`, `ray.put()` 等操作分布式对象的核心 API

---

### 第二步：理解 ray.init() 做了什么

**文件：** `python/ray/_private/worker.py` (搜索 `def init`)

`init()` 函数负责：
1. 启动/连接到 Ray 集群
2. 初始化 Worker 进程
3. 建立与 GCS 和本地 raylet 的连接

核心流程：

```python
def init():
    # 1. 启动本地 raylet 进程（如果不在集群中）
    # 2. 启动 GCS (Global Control Service) 服务
    # 3. 创建 CoreWorker - 这是执行任务的核心组件
    # 4. 初始化对象存储连接
```

---

### 第三步：理解 Task 提交流程

**文件：** `python/ray/remote_function.py` + `python/ray/_private/worker.py`

用户代码：
```python
@ray.remote
def add(a, b):
    return a + b

# 提交任务
result = add.remote(1, 2)
```

执行流程：

```
用户代码: add.remote(1, 2)
    │
    ▼
remote_function.py: RemoteFunction._remote()
    │ 1. 序列化参数
    │ 2. 提交给 CoreWorker
    ▼
worker.py: Worker.core_worker.submit_task()
    │
    ▼
CoreWorker (C++): 发送到本地 Raylet
    │
    ▼
Raylet: 调度任务到 Worker 进程执行
    │
    ▼
返回 ObjectRef 给用户
```

**关键代码位置：**
- `remote_function.py:173-179` - `_remote_proxy` 函数，任务提交的入口
- `worker.py` - `get()` 和 `put()` 函数处理对象

---

### 第四步：理解 Actor 创建和调用

**文件：** `python/ray/actor.py`

用户代码：
```python
@ray.remote
class Counter:
    def __init__(self):
        self.count = 0
    
    @ray.method
    def increment(self):
        self.count += 1
        return self.count

# 创建 Actor
counter = Counter.remote()
# 调用 Actor 方法
result = counter.increment.remote()
```

执行流程：

```
Counter.remote()
    │
    ▼
actor.py: ActorClass._remote()
    │ 1. 创建 Actor
    │ 2. 通过 CoreWorker 提交给 GCS
    ▼
GCS: 管理 Actor 的生命周期和 Placement
    │
    ▼
返回 ActorHandle 给用户

counter.increment.remote()
    │
    ▼
ActorHandle 发送消息到 Actor 所在的进程
```

---

### 第五步：理解 Core Worker (C++)

**文件：** `src/ray/core_worker/`

CoreWorker 是真正执行任务的地方：

```
CoreWorker 主要功能：
1. 任务提交 - submit_task()
2. 任务执行 - ExecuteTask()
3. 对象存储 - put_object(), get_object()
4. 引用计数 - 管理分布式内存
```

**关键文件：**
- `core_worker.cc` - 主实现
- `core_worker.h` - 头文件定义

---

### 第六步：理解 Raylet (节点管理器)

**文件：** `src/ray/raylet/`

每个节点有一个 Raylet：

```
Raylet 功能：
1. 资源管理 - 跟踪 CPU、GPU、内存
2. 本地调度 - 在本节点调度任务
3. Worker Pool - 管理 worker 进程池
4. 对象管理 - 与本地对象存储交互
```

**关键文件：**
- `node_manager.cc` - 节点管理主逻辑
- `scheduler.cc` - 任务调度

---

### 第七步：理解 GCS (全局控制服务)

**文件：** `src/ray/gcs/`

GCS 是集群的大脑：

```
GCS 功能：
1. 节点管理 - 跟踪所有活着的节点
2. Actor 生命周期 - 创建、销毁 actor
3. Job 管理 - 管理 Ray job
4. Placement Group - 资源分组调度
```

**关键文件：**
- `gcs_server.cc` - GCS 主实现
- `actor_info_handler.cc` - Actor 信息处理

---

## 4. 核心文件详解

### 4.1 python/ray/__init__.py

这是学习 Ray 的起点，展示了 Ray 的公共 API。

**关键导出：**
```python
# 核心函数
init, get, put, remote, wait, kill, cancel, shutdown

# Actor 支持
from ray.actor import method

# ID 类型
ActorID, ObjectRef, NodeID, JobID, WorkerID, TaskID, ...
```

### 4.2 python/ray/_private/worker.py

这是 Ray Python 层的核心文件，约 3700 行。

**重要类和函数：**

| 名称 | 行号 | 说明 |
|------|------|------|
| `Worker` 类 | ~500 | 核心 Worker 类 |
| `init()` 函数 | ~2500+ | 初始化 Ray 集群 |
| `get()` 函数 | ~1400+ | 获取远程对象 |
| `put()` 函数 | ~806+ | 存入对象 |
| `wait()` 函数 | ~1600+ | 等待对象就绪 |

**Worker 类核心属性：**
```python
class Worker:
    def __init__(self):
        self.core_worker  # C++ CoreWorker 实例
        self.current_job_id  # 当前 Job ID
        self.function_actor_manager  # 函数/Actor 管理器
        self.serialization_context_map  # 序列化上下文
```

### 4.3 python/ray/_private/services.py

管理 Ray 进程（raylet、GCS、dashboard）。

**关键函数：**

| 函数 | 说明 |
|------|------|
| `start_raylet()` | 启动本地 raylet 进程 |
| `start_gcs_server()` | 启动 GCS 服务 |
| `start_dashboard()` | 启动 Web Dashboard |
| `all_processes()` | 获取所有 Ray 进程信息 |

### 4.4 python/ray/remote_function.py

定义 `@ray.remote` 装饰器。

**核心类：** `RemoteFunction`

**关键方法：**
```python
class RemoteFunction:
    def _remote(self, ...):
        # 1. 序列化参数和函数
        # 2. 通过 CoreWorker 提交任务
        # 3. 返回 ObjectRef
```

### 4.5 python/ray/actor.py

定义 `@ray.remote` class 装饰器（创建 Actor）。

**核心类：** `ActorClass`, `ActorHandle`

**关键方法：**
```python
class ActorClass:
    def _remote(self, ...):
        # 1. 通过 CoreWorker 创建 Actor
        # 2. GCS 分配节点和端口
        # 3. 返回 ActorHandle
```

---

## 5. 组件交互图

```
                    ┌─────────────────────┐
                    │   用户 Python 代码   │
                    └──────────┬──────────┘
                               │ ray.init()
                               ▼
                    ┌─────────────────────┐
                    │   Python Worker      │
                    │  (worker.py)         │
                    └──────────┬──────────┘
                               │
                               ▼
         ┌─────────────────────┴─────────────────────┐
         │                                           │
         ▼                                           ▼
┌─────────────────┐                      ┌─────────────────┐
│  Local Raylet   │                      │   GCS Server    │
│  (C++ raylet/)  │                      │  (C++ gcs/)     │
│                 │                      │                 │
│ - 本地任务调度   │                      │ - 集群状态管理   │
│ - 资源管理      │◄────────────────────►│ - Actor 管理    │
│ - Worker Pool   │     gRPC 通信        │ - Job 管理      │
└────────┬────────┘                      └─────────────────┘
         │                                        │
         │                                        │
         ▼                                        ▼
┌─────────────────┐                      ┌─────────────────┐
│ Object Store    │                      │     Redis       │
│ (Plasma)        │                      │  (状态存储)      │
└─────────────────┘                      └─────────────────┘
```

---

## 6. 通信机制

Ray 使用 **gRPC** 进行进程间通信：

```
Protocol Buffers 定义: src/ray/protobuf/
    ├── common.proto     # 通用消息
    ├── gcs.proto        # GCS 消息
    └── rpc.proto        # RPC 消息

gRPC 服务定义: src/ray/rpc/
    ├── core_worker.grpc
    ├── raylet.grpc
    └── gcs_server.grpc
```

---

## 7. 学习建议

### 推荐的阅读顺序：

.1 **`python/ray/__init__.py`** - 了解 Ray 提供了什么 API
2. **`python/ray/_private/worker.py`** - 核心 Worker 实现
3. **`python/ray/remote_function.py`** - Task 如何提交
4. **`python/ray/actor.py`** - Actor 如何创建
5. **`python/ray/_private/services.py`** - 进程如何启动
6. **`src/ray/core_worker/core_worker.cc`** - C++ 核心执行
7. **`src/ray/raylet/node_manager.cc`** - 任务调度
8. **`src/ray/gcs/gcs_server.cc`** - 集群管理

### 重点关注的函数：

```python
# Python 层
ray.init()              # 启动集群
ray.get()               # 获取对象
ray.put()               # 存入对象
ray.remote()            # 装饰器

# C++ 层
CoreWorker::SubmitTask()   # 提交任务
CoreWorker::ExecuteTask()  # 执行任务
NodeManager::ScheduleTask() # 调度任务
```

---

## 8. 示例：追踪一个 Task 的完整生命周期

假设用户代码是：
```python
import ray
ray.init()

@ray.remote
def add(a, b):
    return a + b

result = add.remote(1, 2)
print(ray.get(result))
```

### 完整流程：

```
1. ray.init()
   └─> worker.py:init()
       ├─> services.py:start_raylet()      # 启动本地 raylet
       ├─> services.py:start_gcs_server()  # 启动 GCS
       └─> Worker().__init__()             # 初始化 Python Worker
           └─> CoreWorker()                # 创建 C++ CoreWorker

2. @ray.remote 装饰
   └─> remote_function.py:RemoteFunction()
       └─> 将普通函数包装为 RemoteFunction 对象

3. add.remote(1, 2)
   └─> remote_function.py:_remote_proxy()
       └─> self._remote()
           ├─> 序列化参数 (1, 2)
           ├─> core_worker.submit_task()
           │   └─> C++ CoreWorker 发送到 Raylet
           └─> 返回 ObjectRef

4. ray.get(result)
   └─> worker.py:get()
       └─> core_worker.get()
           ├─> 本地: 直接从 ObjectStore 读取
           └─> 远程: 通过 gRPC 获取
```

---

## 9. 常用调试技巧

### 查看 Ray 日志位置：
```python
import ray
ray.init()
print(ray._private.utils.get_session_dir())
# 日志在 /tmp/ray/session_xxx/logs/
```

### 打印集群状态：
```python
ray status  # CLI 命令
```

### 查看对象引用：
```python
ray._private.state.object_refs()
```

---

## 10. 总结

Ray 的核心设计：

1. **分层的**：Python API → Core Worker → Raylet/GCS
2. **分布式的**：每个节点运行 Raylet，通过 gRPC 通信
3. **存储分离的**：计算（Raylet）和存储（Object Store）分离
4. **弹性的**：GCS 负责集群级别的协调

**学习重点：**
- `python/ray/_private/worker.py` 是 Python 层的核心
- `src/ray/core_worker/` 是任务执行的核心
- `src/ray/raylet/` 是任务调度的核心
- `src/ray/gcs/` 是集群管理的核心

建议从 `python/ray/__init__.py` 开始，逐步深入到各个组件的源码。
