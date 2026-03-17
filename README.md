# Code Debugging Agent

基于 Agent-R1 框架，使用 GRPO（Group Relative Policy Optimization）强化学习训练 Qwen2.5-7B 模型，使其学会通过多步工具调用（执行代码、搜索错误、生成补丁）自主定位并修复 Python 代码中的 bug。训练数据基于 HumanEval 数据集自动注入 bug 生成。

## 项目结构

```
debug-agent/
├── data/                # 数据层：下载 HumanEval、注入 bug、构建训练数据集
│   ├── download.py      # 下载并缓存 HumanEval 数据集
│   ├── bug_injector.py  # 向正确代码注入 3 类 bug（IndexError / TypeError / LogicError）
│   └── dataset.py       # 封装 DebugDataset，提供 (buggy, correct, tests) 样本
├── tools/               # Agent 工具集：代码执行、错误搜索、代码修补
│   ├── executor.py      # 沙箱执行 Python 代码，返回 stdout/stderr/timeout
│   ├── searcher.py      # 根据错误类型匹配修复建议
│   └── patcher.py       # 验证补丁语法合法性后替换代码
├── env/                 # 强化学习环境：封装单次调试 episode
│   └── debug_env.py     # DebugEnv：管理状态转移、工具调用预算、终止条件
├── reward/              # 奖励函数：基于测试通过率 + 工具调用效率计算 reward
│   └── reward_fn.py     # compute_reward：执行奖励 + 测试奖励 - 调用惩罚 - 超时惩罚
├── train/               # 训练脚本：GRPO 训练循环
│   └── train.py         # 支持 --dry-run 本地验证，GPU 上接入 verl 进行 GRPO 训练
├── eval/                # 评估脚本：baseline vs trained 对比评估
│   └── evaluate.py      # 计算 success_rate / avg_reward / pass@1 等指标
├── demo/                # Gradio 交互演示
│   └── app.py           # Web UI：粘贴 buggy 代码 → 查看调试过程 → 获取修复结果
├── config/              # 配置文件
│   └── config.yaml      # 模型、训练超参、数据分割等配置
└── tests/               # 单元测试
    ├── test_data.py     # 数据加载与 bug 注入测试
    ├── test_tools.py    # 工具函数测试
    ├── test_reward.py   # 奖励函数测试
    └── test_env.py      # 调试环境测试
```

## 本地运行

```bash
# 环境安装
pip install -r requirements-dev.txt

# 运行测试
python -m pytest tests/ -v

# dry-run 验证
python train/train.py --dry-run

# 启动 demo
python demo/app.py
```

## GPU 服务器训练

```bash
pip install -r requirements-gpu.txt
python train/train.py
```

## 评估指标说明

| 指标 | 含义 |
|------|------|
| `success_rate` | 调试成功率——Agent 最终提交的代码通过所有测试用例的比例 |
| `avg_reward` | 平均奖励分数——综合考虑代码执行结果、测试通过数、工具调用次数和超时情况 |
| `avg_tool_calls` | 平均工具调用次数——衡量 Agent 调试效率，越少说明定位问题越精准 |
| `pass@1` | 一次修复成功率——Agent 第一次提交补丁就通过所有测试的比例，衡量修复精度 |

## 技术栈

- **Agent-R1**：强化学习驱动的 Agent 训练框架
- **GRPO**：Group Relative Policy Optimization，无需 critic 网络的高效策略优化算法
- **Qwen2.5-7B**：基座语言模型，作为 Agent 的推理引擎
- **HumanEval**：OpenAI 代码生成基准数据集，用于构建训练和评估数据
- **Gradio**：交互式 Web 演示界面
