# STAT 161 Pokémon 技术决策

截至 2026-11-24。运行证据见 [PROGRESS.md](PROGRESS.md)，操作命令见 [README.md](README.md)。

## 已采用

| 决策 | 依据与实现 |
| --- | --- |
| 课程规则和平台技术文档各自用于对应问题 | 课程 PDF 定义对局、赛制和 LLM 参与要求。官方 Pokémon API 与 Testing 回放用于验证 MCP、合法动作和计时。 |
| 复用 Starter 和 Showdown | `pyproject.toml` 固定 Starter Git 版本，`package-lock.json` 固定 Pokémon Showdown 0.11.11。模拟器沿用 Ace Trainer 设计，用 Python 处理选秀，用 Showdown 执行战斗。 |
| 线上与本地共用玩家视图和联合动作 | `game.py` 统一 `Decision`、招式元数据和合法动作。选秀公开未选卡，道具冲突只影响动作。对手阵容预览和招式揭示来自 Showdown 公开协议。两个槽位不能换到同一名替补，两只倒下且只有一名替补时为一次换入加一次 pass。 |
| 历史卡表作为当前已验证配置 | 49 个官方 18 卡池覆盖历史 48 张全部配置，样本中没有新增 ID 或字段变更。选中卡的简略记录只参与其实际提供字段的比较。用 `--testing-pools` 评估实际观测的池子组合，用 `--cards` 评估指定卡表。 |
| 单方时间银行与策略计时 | 选秀 15 秒、阵容预览 85 秒、战斗决定 50 秒，战斗银行每方 420 秒。模拟器以策略耗时模拟代选和银行判负。课程关于整场计时到期按剩余总 HP 判胜的规则以课程材料为准。 |
| 配对胜率作为策略迭代指标 | 每个种子固定池子、先选席位和 Showdown 种子，交换两个策略的席位。一侧执行失败时该种子的两场都不计入配对胜率。另报告得分率、局速、阶段 P95、回退、代选与判负。 |
| LLM 从战术候选作实际选择 | 战斗候选按预计伤害、击倒机会和出手顺序排序，保留可用的 Protect、换人和控速。DeepSeek V4 Flash 返回合法候选编号，失败或超时采用本地首选。两场官方 Testing 验证了实际模型选择与 MCP 回执延迟。 |
| 阵容预览评分覆盖首发和后排 | 遍历合法四人阵容和首发，让打击覆盖、换入点和控速受益者共同影响排序。模型候选只保留每套阵容和首发组合的一个站位。战斗视图交集取己方队伍与合法换入名单，提供已知的替补 HP、状态和招式。 |

课程规则见 [课程规则 PDF](<参考材料/AI Agent Games_ Rules and Self-hosted Play.pdf>)。接口和复用参考见 [官方 Pokémon API](https://api.altruagent-game.com/skill/pokemon)、[Starter](https://github.com/UCLA-Trustworthy-AI-Lab/altruagent-starter) 和 [Ace Trainer](https://github.com/holmjames/ace-trainer-claude)。

## 后续实验

1. 根据更强对手的输局评估一回合搜索。在保留池子上确认搜索相对战术排序的增益，再生成 BC 示范。
2. 从 BC 出发尝试 PPO，对启发式、冻结快照和其他保留对手自对弈。用胜率与时间成本决定搜索深度、候选数量和是否纳入正式策略。训练与评估设计参考 [VGC-Bench](https://arxiv.org/abs/2506.10326)。
3. 训练机为 Ryzen 9 9950X、RTX 4090 和 48 GB 内存。用 CUDA 训练并保留可跨设备加载的 PyTorch 权重。比赛机为 48 GB M4 Pro MacBook Pro，先测 CPU 和 MPS 的单次延迟、P95 与长局时钟表现，有需要再测 ONNX。
4. 把训练策略或搜索输出作为 LLM 的合法候选，再由 LLM 根据局面参与选择。分别比较启发式、BC、PPO、搜索和 LLM 的增量收益，并在官方 Testing 上复测最终入口。
