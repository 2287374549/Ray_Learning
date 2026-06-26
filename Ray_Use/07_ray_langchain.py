"""
Ray + LangChain 并行 LLM 调用示例

演示内容：
1. 使用 Ray 并行调用多个 LLM 请求
2. 批量处理提示词
3. 使用 Actor 管理 LLM 状态
4. 流式输出处理

注意：这个示例需要安装 langchain 和 openai
pip install langchain langchain-openai
"""

import ray

ray.init()

print("=" * 60)
print("Ray + LangChain 并行 LLM 示例")
print("=" * 60)

# ============================================================
# 示例 1：模拟并行 LLM 调用（无需真实 API key）
# ============================================================
print("\n--- 示例 1：并行文本生成模拟 ---")

@ray.remote
def generate_text_sync(prompt: str, task_id: int) -> dict:
    """
    模拟 LLM 调用

    实际使用时替换为：
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    response = llm.invoke(prompt)
    """
    import time
    import random

    # 模拟 API 调用延迟（200-500ms）
    delay = random.uniform(0.2, 0.5)
    time.sleep(delay)

    # 模拟 LLM 生成的内容
    responses = {
        "什么是 Python?": "Python 是一种高级编程语言，以其简洁易读的语法而闻名。",
        "什么是 Ray?": "Ray 是一个分布式计算引擎，用于扩展 Python 应用。",
        "解释机器学习": "机器学习是人工智能的一个分支，通过数据训练模型进行预测。",
        "什么是深度学习?": "深度学习是使用神经网络的机器学习子领域。",
        "解释强化学习": "强化学习是通过与环境交互学习最优策略的机器学习方法。",
    }

    # 根据提示词返回对应回答（默认返回一个通用回答）
    response = responses.get(prompt, f"关于'{prompt}'的回答：这是模拟的 LLM 响应。")

    return {
        "task_id": task_id,
        "prompt": prompt,
        "response": response,
        "delay": delay
    }


# 定义一群提示词
prompts = [
    "什么是 Python?",
    "什么是 Ray?",
    "解释机器学习",
    "什么是深度学习?",
    "解释强化学习",
    "什么是监督学习?",
    "解释无监督学习",
    "什么是迁移学习?",
]

print(f"准备并行处理 {len(prompts)} 个提示词...")

# 并行调用所有 LLM 请求
start_time = time.time()
futures = [generate_text_sync.remote(prompt, i) for i, prompt in enumerate(prompts)]
results = ray.get(futures)
total_time = time.time() - start_time

print(f"\n完成！总耗时: {total_time:.2f}秒")
print(f"平均每个请求: {total_time/len(prompts):.2f}秒")
print(f"加速比: {(total_time/len(prompts)) * len(prompts) / total_time:.1f}x (理论上串行需要 {sum(r['delay'] for r in results):.2f}秒)")

print("\n结果预览:")
for r in results[:3]:
    print(f"  [{r['task_id']}] {r['prompt'][:20]}... -> {r['response'][:30]}...")

# ============================================================
# 示例 2：使用 Actor 管理 LLM 状态
# ============================================================
print("\n--- 示例 2：带状态的 LLM Orchestrator ---")

@ray.remote
class LLMOrchestrator:
    """
    LLM 编排器 - 管理对话历史和维护上下文

    实际使用中可以：
    - 管理对话历史
    - 实现 token 计数和截断
    - 控制并发请求速率
    """

    def __init__(self, system_prompt: str = "你是一个有帮助的助手。"):
        self.system_prompt = system_prompt
        self.conversation_history = []
        self.request_count = 0

    def chat(self, user_message: str) -> dict:
        """处理单条对话"""
        self.request_count += 1

        # 添加到历史
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # 模拟 LLM 回复
        import time
        time.sleep(0.1)  # 模拟延迟

        response_content = f"我收到了你的消息：'{user_message}'。这是模拟的回复 #{self.request_count}。"

        self.conversation_history.append({
            "role": "assistant",
            "content": response_content
        })

        return {
            "response": response_content,
            "request_count": self.request_count,
            "history_length": len(self.conversation_history)
        }

    def chat_batch(self, messages: list) -> list:
        """批量处理多条消息"""
        results = []
        for msg in messages:
            result = self.chat(msg)
            results.append(result)
        return results

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_requests": self.request_count,
            "history_length": len(self.conversation_history),
            "messages_in_history": len(self.conversation_history) // 2
        }


# 创建 orchestrator
orchestrator = LLMOrchestrator.remote(system_prompt="你是一个友好的 AI 助手。")

# 单个对话
response1 = ray.get(orchestrator.chat.remote("你好！"))
print(f"对话 1: {response1}")

response2 = ray.get(orchestrator.chat.remote("今天天气怎么样？"))
print(f"对话 2: {response2}")

# 批量对话
batch_messages = ["你好吗?", "你喜欢吃什么?", "给我讲个笑话"]
batch_results = ray.get(orchestrator.chat_batch.remote(batch_messages))
print(f"\n批量处理了 {len(batch_messages)} 条消息:")
for r in batch_results:
    print(f"  - {r['response']}")

# 统计信息
stats = ray.get(orchestrator.get_stats.remote())
print(f"\n统计: 总请求数={stats['total_requests']}, 历史消息数={stats['messages_in_history']}")

# ============================================================
# 示例 3：分级处理管道
# ============================================================
print("\n--- 示例 3：LLM 处理管道 ---")

@ray.remote
def classify_intent(text: str) -> str:
    """意图分类"""
    import time
    time.sleep(0.1)

    # 简化分类逻辑
    keywords = {
        "question": ["什么", "如何", "为什么", "who", "what", "how", "why"],
        "request": ["请", "帮我", "能不能", "please", "can you"],
        "greeting": ["你好", "hello", "hi", "早上好"],
    }

    for intent, kws in keywords.items():
        if any(kw in text.lower() for kw in kws):
            return intent
    return "other"


@ray.remote
def generate_response(intent: str, original_text: str) -> str:
    """根据意图生成响应"""
    import time
    time.sleep(0.15)

    templates = {
        "question": "这是一个很好的问题。让我来回答：关于'{text}'...",
        "request": "好的，我会帮你处理'{text}'。",
        "greeting": "你好！很高兴见到你。有什么我可以帮助你的吗？",
        "other": f"我收到了你的消息：'{text}'。",
    }

    return templates.get(intent, templates["other"]).format(text=original_text)


@ray.remote
def polish_response(response: str) -> str:
    """润色响应"""
    import time
    time.sleep(0.05)

    # 添加一些礼貌用语
    polish_templates = [
        "希望我的回答对你有帮助！{}",
        "{} 如果还有其他问题，请随时问我。",
        "{} 祝你有美好的一天！",
    ]

    import random
    template = random.choice(polish_templates)
    return template.format(response)


# 处理一批用户输入
user_inputs = [
    "你好！",
    "什么是机器学习？",
    "请帮我翻译这段话",
    "今天怎么样？",
    "解释一下量子计算",
]

print(f"处理 {len(user_inputs)} 条用户输入...")

# 管道处理：第一阶段 - 意图分类 + 响应生成（并行）
intent_futures = [classify_intent.remote(text) for text in user_inputs]
response_futures = [generate_response.remote(intent, text)
                    for intent, text in zip(ray.get(intent_futures), user_inputs)]

# 第二阶段 - 润色（并行）
polished_futures = [polish_response.remote(resp) for resp in ray.get(response_futures)]
final_responses = ray.get(polished_futures)

# 展示结果
print("\n处理结果:")
for i, (text, response) in enumerate(zip(user_inputs, final_responses)):
    intent = ray.get(intent_futures[i])
    print(f"\n输入: {text}")
    print(f"意图: {intent}")
    print(f"响应: {response}")

print("\n" + "=" * 60)
print("Ray + LangChain 示例完成!")
print("=" * 60)
