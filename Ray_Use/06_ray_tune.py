"""
Ray Tune 超参数搜索示例

演示内容：
1. 定义搜索空间
2. 并行运行超参数实验
3. 使用调度器加速搜索
4. 分析最佳结果
"""

import ray
from ray import tune

ray.init()

print("=" * 60)
print("Ray Tune 超参数搜索示例")
print("=" * 60)

# ============================================================
# 示例 1：简单的网格搜索
# ============================================================
print("\n--- 示例 1：网格搜索 ---")

def objective(config):
    """目标函数：最小化这个值"""
    # 模拟一个学习曲线
    learning_rate = config["lr"]
    batch_size = config["batch_size"]

    # 模拟训练过程
    loss = 0
    for i in range(10):
        # 模拟 loss 下降（带有随机性）
        loss = (1 - learning_rate) * loss + learning_rate * (batch_size / 32) + 0.1 * (0.5 ** i)

    # Ray Tune 会自动追踪这个指标
    tune.report(loss=loss)


# 定义搜索空间
search_space = {
    "lr": tune.loguniform(1e-4, 1e-1),      # 学习率：0.0001 到 0.1 之间
    "batch_size": tune.choice([16, 32, 64]), # 批量大小：选择其中一个
    "hidden_size": tune.choice([64, 128, 256]),  # 隐藏层大小
}

# 运行实验
# num_samples 表示从搜索空间中采样多少组参数
tuner = tune.Tuner(
    objective,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        num_samples=10,  # 只采样 10 组做快速演示
        max_concurrent_trials=2,  # 最多同时运行 2 个实验
    ),
)

results = tuner.fit()

# 分析结果
print(f"\n总共运行了 {len(results)} 个实验")
best_result = results.get_best_result(metric="loss", mode="min")
print(f"最佳配置: {best_result.config}")
print(f"最佳 loss: {best_result.metrics['loss']:.4f}")

# 打印所有实验结果
print("\n所有实验结果（按 loss 排序）:")
sorted_results = sorted(results, key=lambda x: x.metrics["loss"])
for i, r in enumerate(sorted_results[:5]):  # 只显示前 5 个
    print(f"  {i+1}. loss={r.metrics['loss']:.4f}, config={r.config}")

# ============================================================
# 示例 2：使用随机搜索（更高效）
# ============================================================
print("\n--- 示例 2：随机搜索 + Early Stopping ---")

def objective_with_early_stop(config):
    """带 early stopping 的目标函数"""
    import random

    lr = config["lr"]
    momentum = config["momentum"]

    loss = float('inf')
    for step in range(100):
        # 模拟训练：随机波动但总体收敛
        improvement = random.uniform(0.01, 0.1) * lr
        noise = random.uniform(-0.1, 0.1)
        loss = max(0.1, loss - improvement + noise)

        # 每 10 步报告一次
        if step % 10 == 0:
            tune.report(loss=loss, step=step)

        # Early stopping：如果 loss 已经很低，就提前停止
        if loss < 0.2:
            print(f"  Early stop at step {step}, loss={loss:.4f}")
            break


random_search_space = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "momentum": tune.uniform(0.8, 0.99),
    "optimizer": tune.choice(["sgd", "adam", "rmsprop"]),
}

tuner_random = tune.Tuner(
    objective_with_early_stop,
    param_space=random_search_space,
    tune_config=tune.TuneConfig(
        num_samples=15,
        max_concurrent_trials=3,
    ),
)

results_random = tuner_random.fit()

print(f"\n随机搜索完成，运行了 {len(results_random)} 个实验")
best_random = results_random.get_best_result(metric="loss", mode="min")
print(f"最佳 loss: {best_random.metrics['loss']:.4f}")
print(f"最佳配置: {best_random.config}")

# ============================================================
# 示例 3：自定义训练循环
# ============================================================
print("\n--- 示例 3：完整的训练循环搜索 ---")

def train_mnist(config):
    """
    更真实训练场景：
    - 有真正的训练/验证划分
    - 保存检查点
    - 使用 early stopping
    """
    import random
    random.seed(42)

    # 模拟超参数
    lr = config["lr"]
    dropout = config["dropout"]
    hidden_size = config["hidden_size"]

    # 模拟训练
    train_loss_history = []
    val_acc_history = []

    best_val_acc = 0
    patience = 3  # 早停耐心值
    patience_counter = 0

    for epoch in range(20):
        # 模拟训练 loss
        train_loss = 2.0 * (0.9 ** epoch) * (1 + random.uniform(-0.1, 0.1)) + dropout * 0.1
        train_loss_history.append(train_loss)

        # 模拟验证准确率
        val_acc = min(0.95, 1.0 - train_loss * 0.3 + random.uniform(-0.05, 0.05))
        val_acc_history.append(val_acc)

        # 保存最佳准确率
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # 报告指标
        tune.report(
            train_loss=train_loss,
            val_acc=val_acc,
            best_val_acc=best_val_acc,
            epoch=epoch,
        )

        # Early stopping
        if epoch > 0 and val_acc < val_acc_history[-2]:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break
        else:
            patience_counter = 0


mnist_search_space = {
    "lr": tune.loguniform(1e-4, 1e-2),
    "dropout": tune.uniform(0.1, 0.5),
    "hidden_size": tune.choice([128, 256, 512]),
    "optimizer": tune.choice(["adam", "sgd"]),
}

tuner_mnist = tune.Tuner(
    train_mnist,
    param_space=mnist_search_space,
    tune_config=tune.TuneConfig(
        num_samples=12,
        max_concurrent_trials=4,
    ),
)

results_mnist = tuner_mnist.fit()

# 展示结果
print(f"\nMNIST 搜索完成，运行了 {len(results_mnist)} 个实验")

# 按验证准确率排序
sorted_mnist = sorted(results_mnist, key=lambda x: x.metrics["val_acc"], reverse=True)

print("\n前 3 名配置:")
for i, result in enumerate(sorted_mnist[:3]):
    print(f"\n  #{i+1}: val_acc={result.metrics['val_acc']:.4f}")
    print(f"     配置: lr={result.config['lr']:.6f}, "
          f"dropout={result.config['dropout']:.2f}, "
          f"hidden_size={result.config['hidden_size']}, "
          f"optimizer={result.config['optimizer']}")

# 最佳结果统计
best_mnist = sorted_mnist[0]
print(f"\n最佳验证准确率: {best_mnist.metrics['val_acc']:.4f}")
print(f"最终训练 loss: {best_mnist.metrics['train_loss']:.4f}")

print("\n" + "=" * 60)
print("Ray Tune 示例完成!")
print("=" * 60)
