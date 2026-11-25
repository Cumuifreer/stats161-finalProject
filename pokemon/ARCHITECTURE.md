# Pokémon 双打架构对比与搜索原型

本项目的长期方向是用轻量 Policy 提议合法联合动作，用 Showdown 对少量候选和可能的对手回应做一回合搜索，再以 Value 估计后续胜率。Joint 是当前的对战基线、候选先验和离线搜索对照。正式入口继续让 LLM 根据当前可见局面实质参与选招。

## BC v2 的有限对照

三组 CUDA 训练各用同样的 320 局、2,656 条非强制 Joint 示范、相同的 64 宽 MLP 和 12 轮训练。硬标签以 Joint 的首选计算交叉熵，软标签采用 Joint 候选分数的 softmax，温度固定为 0.25。历史及固定卡能力值与候选限制分别对照。682 条模仿检查在另一组 40 个选秀池上完成。其中 Joint 与 Tactical 的首选有 309 次不同，有 373 次一致。

| 策略 | 模仿 Top‑1 | 跟对 Joint 改选 | 改错原先一致的动作 | 冻结对局胜率 |
| --- | ---: | ---: | ---: | ---: |
| v1 全合法动作 | 55.13% | 71 | 68 | 277/640，43.28% |
| v1 同权重限制候选 | 55.13% | 71 | 68 | 277/640，43.28% |
| shortlist-hard | 52.79% | 75 | 88 | 277/640，43.28% |
| history-hard | 58.06% | 120 | 97 | 283/640，44.22% |
| history-soft | 56.16% | 145 | 135 | 284/640，44.38% |
| Joint 对照 |  |  |  | 329/640，51.41% |

五个对手各 64 个换座对。冻结清单固定了种子、池指纹和 Ace Trainer revision。`tactical-calibrated` 未参与生成示范。套件总计 3,840 局，零非法动作、自动代选、计时判负和执行错误。`history-soft` 相对 v1 平均增加 1.09 个百分点，按种子配对 bootstrap 的 95% 区间为 −3.59 至 +6.09。对 Joint 控制的平均差为 −7.03 个百分点，95% 区间为 −11.56 至 −2.66。`history-hard` 对 Joint 的直接胜场从 v1 的 42/128 提高到 56/128，对 Pivot 则从 53/128 降到 44/128。下一轮模型选择应使用新留出池，当前冻结套件可继续作回归对照。 [冻结清单](data/frozen_benchmark.json)

限制 v1 候选后，这一批 682 个局面未改变选招，冻结赛逐对得分一致。Windows CPU 决策 P95 从 1.90–1.97 ms 降到 0.69–0.82 ms，P99 从 2.32–2.36 ms 降到 0.89–0.95 ms。`history-soft` 的 P95 为 0.73–0.83 ms、P99 为 0.84–1.03 ms。历史模型有 32,321 个参数，权重 129.6 KiB。CUDA 优化器阶段每组不到一秒。当前不足主要体现在对局质量，增加同目标的计算预算没有形成明确晋级信号。

## 四个实现的比较

### VGC-Bench

- **状态与网络**。论文用 12 个 Pokémon 实体表示两队，另加聚合 token。审阅版本 `d79f953` 的实现将天气、场地、墙等上下文与每个实体的 HP、能力变化、状态、效果和招式属性拼接。道具、特性和招式各有嵌入，Policy 和 Value 使用独立的特征提取器。训练入口的模型宽度为 256，实体编码器有 3 层、4 个注意力头。论文另讨论可选的时间维 Transformer。 [论文](https://arxiv.org/html/2506.10326v3#S4) · [状态编码](https://github.com/cameronangliss/VGC-Bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/vgc_bench/src/policy_player.py#L264) · [网络](https://github.com/cameronangliss/VGC-Bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/vgc_bench/src/policy.py)
- **动作与 BC**。每槽用固定动作词表，包括目标、换人和赛制特殊机制。采样槽 A 后再更新槽 B mask，消除重复换人等非法组合。Replay 预处理从 OTS 日志回放轨迹，对被击倒前未执行的命令和未全部揭示的后排作专门处理。BC loader 用合成 mask 处理日志中缺失的完整合法列表。我们有决策时真实的 `Decision.actions`，可以直接学习完整联合动作分布并精确掩掉 padding。 [动作掩码](https://github.com/cameronangliss/VGC-Bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/vgc_bench/src/policy.py#L81) · [回放重建](https://github.com/cameronangliss/VGC-Bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/vgc_bench/logs2trajs.py#L84) · [BC loader](https://github.com/cameronangliss/VGC-Bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/vgc_bench/pretrain.py#L31)
- **PPO、Fictitious Play 与评价**。PPO 使用胜、平、负的终局奖励。纯 Self-Play、均匀抽历史 checkpoint 的 Fictitious Play、按历史交手矩阵 Nash 分布抽样的 Double Oracle 共用训练框架。论文用交叉对阵、Alpha-Rank、未见队伍和独立 exploiter 测不同失效方式。BC 初始化的 PPO 组合在各队伍规模中整体排在前列，同时多队伍泛化依然困难。适合迁移的是冻结对手池和跨阵容评估。107 个每槽动作、太晶化、公共天梯队伍构造和 OTS 重建假定在课程赛制中用处有限。 [训练](https://github.com/cameronangliss/VGC-Bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/vgc_bench/train.py) · [对手采样](https://github.com/cameronangliss/VGC-Bench/blob/d79f9532947ac114dce1dda2456a590afcd375b2/vgc_bench/src/callback.py#L258) · [泛化与交叉对阵](https://arxiv.org/html/2506.10326v3#S5)

### Ai-vgc

- **结构与特征**。`671f1d6` 的 Model17 主干用 17 个 token 表示 8 个已带入 Pokémon、8 个己方场上招式及场地，默认 8 层、宽度 256、8 个注意力头。招式分数由行动者、招式和目标的上下文共同产生。Model18 加独立的全六只预览编码和历史上下文，可重用主干输出为槽 B 打分。Model19 注入公开证据得到的配置假设和伤害／出手顺序不确定性，部分版本增加对手配置预测头。适合我们的是复用实体和招式计算、共享编码后输出动作与价值。约 750 万参数的整套 Model17 及 Mega 动作头不适合直接搬入首次原型。 [Model17](https://github.com/Nisanth-Nimashakavi/Ai-vgc/blob/671f1d63b585cb14be670ebc93af4e5e1b60cd1d/src/ai_vgc/nn/model17.py) · [Model18](https://github.com/Nisanth-Nimashakavi/Ai-vgc/blob/671f1d63b585cb14be670ebc93af4e5e1b60cd1d/src/ai_vgc/nn/model18.py) · [Model19](https://github.com/Nisanth-Nimashakavi/Ai-vgc/blob/671f1d63b585cb14be670ebc93af4e5e1b60cd1d/src/ai_vgc/nn/model19.py)
- **学习与评估**。回放数据有动作、终局结果与对手动作等监督。闭队表分支在输入端移除对手队表或遮蔽字段，蒸馏代码对两槽的合法分布分别监督，槽 B 受槽 A 条件限制。PPO 实现含 GAE、限制更新幅度和辅助目标，另外有按对历史 checkpoint 失利情况优先采样的 PFSP 路径。评估脚本分别验证原生环境、Showdown 影子状态和对手组合。我们可借鉴它对可见信息、分支、训练标签的隔离。固定 48 卡已知配置降低了使用大众天梯道具和 EV 边缘分布的价值，剩余的不确定性集中在后排、回合状态和对手策略。 [数据](https://github.com/Nisanth-Nimashakavi/Ai-vgc/blob/671f1d63b585cb14be670ebc93af4e5e1b60cd1d/src/ai_vgc/nn/dataset.py) · [闭队表蒸馏](https://github.com/Nisanth-Nimashakavi/Ai-vgc/blob/671f1d63b585cb14be670ebc93af4e5e1b60cd1d/src/ai_vgc/nn/cts_distill.py) · [PPO](https://github.com/Nisanth-Nimashakavi/Ai-vgc/blob/671f1d63b585cb14be670ebc93af4e5e1b60cd1d/src/ai_vgc/nn/rl18.py) · [PFSP 测试](https://github.com/Nisanth-Nimashakavi/Ai-vgc/blob/671f1d63b585cb14be670ebc93af4e5e1b60cd1d/tests/test_native_pfsp.py) · [Showdown 对照](https://github.com/Nisanth-Nimashakavi/Ai-vgc/blob/671f1d63b585cb14be670ebc93af4e5e1b60cd1d/scripts/eval_showdown_stateful_shadow.py)
- **搜索**。代码能为多个假设世界建行动收益矩阵，并约束己方策略跨世界共享。对手在每个世界可以有自己合法的回应。矩阵求解中的 Fictitious Play 是 LP 的迭代备选，与训练期间抽历史 checkpoint 的 Fictitious Play 是两个用法。随机性可在每个世界内先按多个 seed 平均。仓库也有显式伤害骰、暴击和二次效果枚举，但该文件的完整 chance 结算仍有待实现步骤。我们应直接让 Showdown 处理效果和随机性。 [矩阵求解](https://github.com/Nisanth-Nimashakavi/Ai-vgc/blob/671f1d63b585cb14be670ebc93af4e5e1b60cd1d/src/ai_vgc/nn/double_oracle.py) · [公开视角续搜](https://github.com/Nisanth-Nimashakavi/Ai-vgc/blob/671f1d63b585cb14be670ebc93af4e5e1b60cd1d/src/ai_vgc/nn/public_depth2.py) · [随机事件分支实现](https://github.com/Nisanth-Nimashakavi/Ai-vgc/blob/671f1d63b585cb14be670ebc93af4e5e1b60cd1d/src/ai_vgc/nn/chance_branching.py#L188)

### VGC-Pokemon-Showdown-AI

- **模型与学习**。`56f6710` 的项目扩展了 VGC-Bench 的双打 actor-critic 和两槽 mask。其 `AnchoredPPO` 在 PPO 上增加冻结起点策略的 `KL(anchor || policy)`，按观测 KL 自适应调整权重。价值搜索同时有 PPO critic 叶节点以及基于终局标签训练的独立 Value 路径，可用相同枝叶测校准与效果。对手预测器从公开状态训练招式／换人与目标分支，然后过滤不合法搭配。联合回应支持两槽概率乘积和额外耦合校准。可迁移的是以局为单位的数据隔离、价值校准和可记录的对手回应覆盖率。其 Reg M-B、Mega 和天梯先验依赖其赛制。 [Policy](https://github.com/philmantatsky/VGC-Pokemon-Showdown-AI/blob/56f671086c18c585617e14c90dcccf30c3ae5bcd/vgc_bench/src/policy.py) · [Anchored PPO](https://github.com/philmantatsky/VGC-Pokemon-Showdown-AI/blob/56f671086c18c585617e14c90dcccf30c3ae5bcd/vgc_bench/src/anchored_ppo.py) · [叶节点价值](https://github.com/philmantatsky/VGC-Pokemon-Showdown-AI/blob/56f671086c18c585617e14c90dcccf30c3ae5bcd/vgc_bench/src/outcome_value.py) · [联合对手回应](https://github.com/philmantatsky/VGC-Pokemon-Showdown-AI/blob/56f671086c18c585617e14c90dcccf30c3ae5bcd/vgc_bench/src/oppmodel/joint.py)
- **最接近本项目的搜索参考**。桥接代码用 Showdown 序列化状态的分支副本、独立 RNG seed 和同一个引擎结算双方命令。线上求解另维护由公开输入更新的影子世界。一回合同时行动矩阵可用普通零和求解或以双方策略为先验的正则化求解。评估把预算用尽、实际覆盖的回应、覆盖掉的原选和未见队伍分开统计。源码中特别指出了无正则的微小收益导致过多改选的实测问题。我们应先检查相对 Joint 的配对收益和不确定性，再采用更复杂的矩阵求解。 [Showdown 桥](https://github.com/philmantatsky/VGC-Pokemon-Showdown-AI/blob/56f671086c18c585617e14c90dcccf30c3ae5bcd/tools/exact_showdown_bridge.js#L63) · [公开快照](https://github.com/philmantatsky/VGC-Pokemon-Showdown-AI/blob/56f671086c18c585617e14c90dcccf30c3ae5bcd/vgc_bench/src/live_snapshot.py) · [同时行动矩阵](https://github.com/philmantatsky/VGC-Pokemon-Showdown-AI/blob/56f671086c18c585617e14c90dcccf30c3ae5bcd/vgc_bench/src/exact_planner.py#L1193) · [搜索与未见队伍评估](https://github.com/philmantatsky/VGC-Pokemon-Showdown-AI/blob/56f671086c18c585617e14c90dcccf30c3ae5bcd/evaluation/eval_heldout.py)

### PokéChamp

- **ICML 2025 方法**。LLM 为限深 minimax 提供我方候选、对方可能动作和叶节点评价。代码结合当前局面、历史、伤害与速度估计及公开用量先验推断对方配置。常规搜索代码对双方候选做组合，在 `LocalSim` 中用近似转移做前推，再做 max-min 回传，部分路径有快速价值或伤害回退。主论文证据来自单打，且作者报告了应对消耗与反复换人时的弱点。仓库另有 VGC LLM 入口，论文的单打胜率不能外推到本课程双打。 [ICML 论文](https://proceedings.mlr.press/v267/karten25a.html) · [实现](https://github.com/sethkarten/pokechamp/blob/f5c359e166625bc1d2955aeaf9c90188b1243550/pokechamp/llm_player.py#L693) · [近似转移](https://github.com/sethkarten/pokechamp/blob/f5c359e166625bc1d2955aeaf9c90188b1243550/poke_env/player/local_simulation.py#L415) · [VGC 入口](https://github.com/sethkarten/pokechamp/blob/f5c359e166625bc1d2955aeaf9c90188b1243550/pokechamp/llm_vgc_player.py)
- **取舍**。我们应保留候选裁剪、对手建模和叶节点评估三个可独立比较的模块。正式 Agent 让 LLM 在带有搜索证据的候选之间真实选招。模拟分支批量运行本地小模型，可以控制 P99 和付费调用次数。逐叶调用 LLM、单打 `LocalSim` 的伤害重写和纯最坏回应求解都应先留在对照位置。

## 中盘恢复的证据与信息边界

本地 `restore-audit` 在两个同种子 Showdown 进程提交相同命令，每次中盘把其中一边的引擎对象做 `toJSON → fromJSON → restart`，然后比较两边玩家视图、合法模板和后续终局。32 局的 401 个中盘检查点均一致，其中 79 个为替换请求，32 个终局完全一致。恢复请求含进程通信的 P95 为 1.46 ms、P99 为 1.79 ms。完整状态留在测试进程内。这验证了模拟器的序列化路径，支持未来在**自行构造的假设世界**上做反事实分支。 [审计实现](pokemon_agent/restoration.py) · [Showdown 桥](pokemon_agent/showdown.cjs)

课程平台的真实决策只有本座位视角。中盘恢复必须面对对手带入但未露面的队员、精确 HP 舍入、PP 及 Choice 锁、效果剩余时长、刚上场状态和条件触发次序等不确定性。现有 `Decision` 主要保留名字、当前公开 HP／状态和合法模板。相同己方观测可以对应不同的对手合法换人。官方 Testing 的 `parity` 到首回合，后续记录可用于逐帧合法性与公开历史审计。下一步从授权的原始观察提取实际存在的字段，允许在合法可见证据充分的局面构造多个候选世界，其余局面继续用 Joint。 [官方 Pokémon 观察接口](https://platform.altruagent-game.com/skill/pokemon) · [Decision](pokemon_agent/game.py) · [当前 parity](pokemon_agent/parity.py)

## Showdown 一回合搜索实测

`bench --left search --battle-only` 在本地启用独立的 Showdown 搜索进程。它从公开完成事件、当前己方 `Decision`、己方公开给运行者的 PP 和固定卡表重建局面。对手后排必须四只均已露面，场地效果、复杂持续状态和无法匹配的己方合法动作都会触发 Joint 回退。对手 HP 按公开显示范围取三个兼容样本。分支不会访问主对局对象、真实随机数状态或尚未公开的对手动作。 [决策逻辑](pokemon_agent/search.py) · [搜索进程](pokemon_agent/search.cjs)

每次搜索最多比较 4 个己方动作、3 个对手回应和 3 组共同 RNG seed。Joint 首选始终参评，并试一个战术排序的候选集外动作。Showdown 结算到下一常规行动或终局，包括其间的默认强制替换。叶节点用终局、存活、HP 和出手速度差评分。相对增益达到 0.12 分且至少两组采样支持改选时才替换 Joint。250 ms 搜索预算内未完成的矩阵不用于改选。

| 冻结对手 | Joint 胜场 | Joint + Search 胜场 |
| --- | ---: | ---: |
| Joint | 64/128 | 65/128 |
| Tactical | 74/128 | 75/128 |
| Pivot | 60/128 | 61/128 |
| Pivot-base | 57/128 | 58/128 |
| Tactical-calibrated | 74/128 | 74/128 |
| 合计 | 329/640 | 333/640 |

同池同种子换座的 1,280 局对照中，Search 将 6 局负局变为胜局，也将 2 局胜局变为负局。平均得分差为 **+0.63 个百分点**，95% 配对区间为 **−0.16 至 +1.88**。336 个完成重建的局面全部匹配己方合法动作，331 个完成整批搜索，共 11,916 条分支。6,041 次战斗决策中覆盖率为 5.48%，改选 89 次，其中 6 次选中候选集之外的动作。5 次触及搜索预算并回退，正式决策时钟模拟的自动代选与判负均为零。Windows 已完成搜索的 P95 为 55.59–75.58 ms、P99 为 64.85–92.36 ms，各范围取五个对手的统计结果。

另取捕获自官方 Testing 的 52 个选秀池，使用从 98601 起的新随机种子在本地复核。两种策略对 Joint 均为 52/104，对 Pivot 均为 42/104，416 局全部完成。两组都没有观测到净胜场增益。

这组结果支持保留本地原型。现有冻结套件上增益区间跨零，继续用 Joint 作为基线，暂缓用搜索标签训练。最值得增加的覆盖是有明确公开开始和结束证据的 Tailwind、Trick Room 与天气。冻结对照中有 1,694 次决策在通过阵容揭示门槛后因场地效果回退。之后可用新的留出对局判断覆盖变大是否能转化为胜率。官方入口继续使用原 LLM 策略，原型使用的是本地 Showdown 的公开协议事件。

## Search v2 最终验证

搜索从公开 `-sidestart`、`-fieldstart`、`-weather` 及对应结束事件恢复 Tailwind、Trick Room 和限时天气。以 `upkeep` 推进计数，天气的 `[upkeep]` 提示保留原开始时间。固定卡配置和已公开的出场及道具事件支持恢复天气岩石。分支样本验证了潮湿岩石降雨的计时、Trick Room 再次使用时的解除、Tailwind 到期，以及缺少天气开始事件时回退。

新保留集从 `100001` 开始，共 64 个池。对五个固定对手换座，Search v2 与 Joint 各跑 640 局，胜场分别为 331 和 324。Search 对 Joint 直接对阵为 64/128。按种子配对 bootstrap 的平均得分差为 +1.09 个百分点，95% 区间为 −0.31 至 +2.66。6,022 次战斗决策完成 756 次合法搜索，覆盖率 12.55%，其中 299 次带有恢复的场地效果。所有完成搜索都匹配当前合法动作集。Search 战斗决策的 Windows P95 为 47.62–56.77 ms，P99 为 74.85–81.02 ms。两组的非法动作和超时次数均为零。这个结果保留 Joint 为对战基线，Search v2 作为离线对照冻结。 [保留集](data/search_v2_benchmark.json)

## Policy/Value PPO v1

PPO 从兼容的 `history-hard` BC v2 初始化，保留 64 宽的原状态编码与 Policy 权重，新增共享状态表示的 Value 头。Policy 使用 `Decision.actions` 中全部合法联合动作，批处理掩码只移除补齐行。Value 以当前席位的胜 `+1`、平 `0`、负 `−1` 为终局目标，GAE 的终局 bootstrap 为零。对手按整局冻结，包括 Joint、Pivot、Pivot-base 和历史快照。六轮的历史策略对局比例由 10/96 增到 34/96。训练目标和历史对手池借鉴 VGC-Bench，回报与 PPO 裁剪验证参考 Ai-vgc。 [VGC-Bench](https://arxiv.org/abs/2506.10326) · [Ai-vgc PPO](https://github.com/Nisanth-Nimashakavi/Ai-vgc/blob/671f1d63b585cb14be670ebc93af4e5e1b60cd1d/src/ai_vgc/nn/rl18.py)

训练使用 288 个独立池的 576 局和 6,864 个学习席位决策。Ryzen 9 9950X 上六个采样进程共用 21.78 秒，热启动中位吞吐量为 34.2 局／秒。RTX 4090 上六轮更新合计 1.48 秒，热启动中位吞吐量为 24,231 样本／秒，包含每轮四遍 PPO 更新。PyTorch CUDA 分配峰值为 228.48 MiB。进程重启后恢复了第 2 轮的优化器和历史快照，从同一个 checkpoint 独立重跑第 3 轮得到相同的 1,087 个决策样本和逐元素完全一致的权重。第 2、4、6 轮分别用独立开发池对 Joint 打 24 局，胜场为 10、9、10。

最终保留集固定使用种子 `130001` 起的 64 个池。每一行每个策略打 128 局，包含双方换座。`tactical-calibrated` 没有参与 PPO 采样及 BC v2 示范。

| 对手 | Joint 对照 | PPO 更新前，全合法动作 | PPO 第 6 轮 |
| --- | ---: | ---: | ---: |
| Joint | 64 | 31 | 34 |
| Tactical | 77 | 47 | 47 |
| Pivot | 59 | 39 | 37 |
| Pivot-base | 56 | 37 | 44 |
| Tactical-calibrated | 77 | 43 | 44 |
| 合计 | 333/640 | 197/640 | 206/640 |

PPO 对 Joint 的直接胜率为 26.56%。对五种对手平均得分相对 Joint 对照为 −19.84 个百分点，按种子配对的 95% 区间为 −24.84 至 −14.84。相对同动作集合的初始化为 +1.41 个百分点，95% 区间为 −1.72 至 +4.69。1,920 局没有非法动作、超时或执行错误。PPO 战斗决策在 Windows CPU 上 P95 为 2.00–2.15 ms，P99 为 2.32–2.51 ms。模型共有 36,546 个参数，最终 `state_dict.pt` 为 147.18 KiB。六个可移植样本已用 CPU 和 CUDA 核对合法首选、Policy 分数及 Value。M4 Pro 的 CPU 与 MPS 延迟仍需实机测量。正式比赛继续使用现有 Joint + DeepSeek 入口。 [独立保留集](data/ppo_benchmark.json) · [复现命令](README.md#battle-ppo)

追加的同权重对照将 BC 恢复为训练时的 Tactical 候选集。它对 Joint 拿到 57/128，五对手合计 255/640。相对全合法动作的初始化提高 9.06 个百分点，95% 配对区间为 +4.53 至 +13.44。这个诊断表明，扩大候选集会显著压低当前 BC 权重的冷启动表现。该诊断没有用于选择 checkpoint。

## PPO v2 候选动作诊断

Joint 教师只在 Tactical 候选内给出标签和排序。BC v1 的交叉熵在全合法集上归一化，候选外动作参与分母。`shortlist-hard` 和 `history-hard` 训练时只保留 Tactical 候选。PPO v1 从后者初始化时已把 Rollout、旧 log probability、更新掩码与推理一起切换到全合法集。PPO v2 把候选模式保存在模型配置中，由 Rollout、推理、冻结对手和便携验证共用。PPO 更新使用 Rollout 的动作行和动态掩码，并在第一次梯度更新前逐行核对旧 log probability。唯一合法动作直接提交，概率为 1，仍提供 Value 目标。Tactical 对合法列表至少保留首选。

诊断使用从 `138001` 开始的 64 个独立池。固定 Joint 轨迹包含 1,006 个非强制决策和 84 个强制决策。所有 Joint 标签及排序动作都在候选中，平均合法动作数为 70.20，候选数为 5.48。同权重切换模式时，候选内 logits 最大绝对差为 `7.63e-6`。

| BC 权重 | 特征版本 | 全合法首选在候选外 | 候选外概率质量均值 | 教师 Top-1 全合法 → 候选 | 对 Joint 胜场全合法 → 候选 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 | 1 | 0/1,006 | 0.08% | 566 → 566 | 50 → 50 |
| shortlist-hard | 1 | 103/1,006 | 26.66% | 518 → 570 | 51 → 51 |
| history-hard | 2 | 280/1,006 | 33.77% | 470 → 612 | 41 → 53 |

同一权重与特征的对照，固定了棋局种子、卡池、对手和换座顺序。`history-hard` 候选集的配对增益为 +9.38 个百分点，95% 区间为 +0.78 至 +17.97。`shortlist-hard` 在特征 v1 上也出现候选外优先，表明这种现象无须切换到历史特征才会发生。它的净胜场未改善，也说明越界动作减少与胜率改善之间仍需完整对局验证。

## PPO v2 短训练与盲测

固定 `history-hard` 初始化、状态和网络、CUDA 更新参数，以及训练卡池。两组各四轮，每轮 32 个换座对。递增组的历史对手每轮分别占 6、12、18 和 22 局，常量组各轮占 6 局。其余对局继续混合 Joint、Pivot 和 Pivot-base。训练池从 `140001` 开始，开发池从 `150001` 开始。独立盲测池从 `160001` 开始，另保留 `170001–170064` 给后续盲测。本轮未用后者跑对局。正式入口和 Search v2 继续冻结。

两组共 512 局，收集 5,041 个学习席位决策。合法候选数量只出现 1、2、4、5 和 6，368 次唯一动作的采样概率为 1。更新前重算旧 log probability 的最大绝对误差为 `4.77e-6`。递增组约为 30.5 局／秒（复用采样进程的第 2 和第 4 轮），四轮 CUDA 更新共 0.44 秒，分配峰值为 22.65 MiB。平均熵为 0.460–0.512，近似 KL 为 0.0025–0.0055，裁剪比例为 3.35–7.65%。递增组从第 2 轮的 checkpoint 恢复。递增组第 2 与第 4 轮在开发集对 Joint 都为 23/48，常量组分别为 20/48 和 19/48。

| 盲测对手 | Joint 对照 | BC 初始化 | 递增第 2 轮 | 递增第 4 轮 | 常量第 4 轮 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint | 64 | 53 | 50 | 53 | 52 |
| Tactical | 73 | 60 | 58 | 60 | 61 |
| Pivot | 67 | 48 | 43 | 47 | 51 |
| Pivot-base | 67 | 43 | 43 | 44 | 46 |
| Tactical-calibrated | 75 | 56 | 55 | 56 | 58 |
| 冻结 PPO v1 第 6 轮 | 89 | 65 | 69 | 66 | 67 |
| **合计** | **435/768** | **325/768** | **318/768** | **326/768** | **335/768** |

各行每组 128 局，使用新池和双方换座。相对 BC，递增第 4 轮的平均胜率变化为 +0.13 个百分点，95% 配对区间为 −1.95 至 +2.34，常量第 4 轮为 +1.30 个百分点，区间为 −0.91 至 +3.65。递增第 4 轮与 Joint 对照的平均差为 −14.19 个百分点，区间为 −19.53 至 −8.85。两组均未达到扩大训练的条件。3,840 局无非法动作、超时或执行错误。递增第 4 轮的 Windows CPU 战斗决策 P95 为 0.71–0.81 ms，P99 为 0.85–0.99 ms。 [盲测清单](data/ppo_v2_benchmark.json)

递增第 4 轮的开局 Value 与实际终局回报在各对手上的 Pearson r 为 0.15–0.34，六类对手平均 MSE 为 0.965。以每个对手实际平均回报作为常数参考，MSE 为 0.964。Value 有弱排序信号，校准改善尚未成立。两组最终 `state_dict.pt` 均为 147.18 KiB，CPU 与 CUDA 便携样本对照均通过。

固定诊断轨迹上，相对初始化，递增第 2、4 轮和常量第 4 轮分别改选 54、65 和 69 次。递增第 4 轮的平均 `KL(BC || PPO)` 为 0.0215，教师首选命中从 612 降到 603。动作分布确实发生了变化，本轮的配对成绩未支持直接增加 PPO 更新次数。

## 自主研发实验

BC v1/v2 的全局拟合把 Joint 的逐局判断压进同一个小模型。候选支持修复消除了部分冷启动损失，PPO v2 的两种历史对手混合在 768 局独立评估中都没有可靠地改善初始化。Search v2 每个战斗决策都能取得的公开证据有限，实际完成整矩阵的覆盖率为 12.55%。当前最值得验证的是保留 Joint 作为默认决策，直接衡量能够改善完整对局胜率的改选。

本轮复读了 Ai-vgc 的 `search.py`，其己方候选、对手候选、共同 RNG 和价值输出分别受控，而且训练日志报告了没有对手预测时的搜索下降。VGC-Bench 在固定阵容上的强结果，伴随着跨阵容扩展时的泛化成本。Foul Play 的转换代码区分已知与估计状态，积分数个可能世界的 MCTS 证据主要来自单打。Metamon 的离线 Transformer 同样有单打和大量历史数据前提。基于这些成本差异，本轮先拟合少量站位和回应参数，随后用完整游戏结果训练局部改选。 [Ai-vgc 搜索](https://raw.githubusercontent.com/Nisanth-Nimashakavi/Ai-vgc/main/src/ai_vgc/nn/search.py) · [Ai-vgc 实验日志](https://github.com/Nisanth-Nimashakavi/Ai-vgc/blob/main/docs/training-ideas.md) · [VGC-Bench](https://arxiv.org/html/2506.10326v3) · [Foul Play 状态转换](https://raw.githubusercontent.com/pmariglia/foul-play/main/fp/search/poke_engine_helpers.py) · [Metamon 论文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_340.pdf)

### 站位威胁和对手回应

假设 Joint 的一轮交换分数遗漏了存活行动态势。纯 CPU 拟合将原候选与扩展候选、替代攻击目标及 Protect、下一回合高威胁位置做组合比较。六种改动与 Joint 在四个对手上各打 32 个换座对，合计 1,792 局。开发集选中原六候选加 `0.55` 的下一回合站位分，成绩为 146/256，Joint 为 130/256。对 `170001` 起的 64 个独立卡池冻结后，六对手的平均配对差为 −0.65 个百分点，95% 区间 −4.04 至 +2.73，对 Joint 直接为 62/128。两臂 1,536 局无非法动作或超时。模型 Windows CPU 战斗 P95 为 1.22–1.26 ms，P99 为 1.34–1.52 ms。实验停止，拟合结果与权重仅作复现。

### 用终局反事实训练局部改选

假设完整对局的配对胜负比短期 HP 更适合作动作标签。沿用 Joint 完成同种子前缀，在第 1–3 回合之一分别试前三候选，然后交回 Joint。每次确认三条分支拥有相同玩家视角特征，整个流程不序列化引擎内部状态。对手轮换 Joint、Tactical、Pivot 和 Pivot-base。用 512 个训练局面及 128 个独立验证局面，每局面三条完整对局，共 1,920 局。模型为 73,473 参数的 PyTorch 候选分数网络，六进程采集耗时约 23.82 秒，CUDA 更新约 0.30 秒。验证的 56 个有胜负差异候选中，方向准确率为 44.64%。相对收益 MSE 为 0.2217，预测恒为无收益的参考为 0.2188。采用 0.10 改选阈值时，四对手共 256 局全部保留 Joint 首选。两臂胜场均为 131，停止扩大训练。

### 公开阵容回放加 Joint

对局早期的高价值决策在已知双方六卡的阵容预览。对手实际带入仍未知，使用 Tactical 与朴素启发式产生两种可能阵容。己方候选保留最多五个合法阵容，优先比较不同带入组合。每个行动对用两组共同 RNG 跑完游戏，通过已有 Showdown 与 Joint 决策，单次预览最多 20 局。只有候选合计多赢至少两局且对两种假设分别没有下降才改选。五秒评估预算不足或公共卡不完整时退回 Tactical 首选。Ai-vgc 的预览矩阵代码在它自己的学习式预览上未获得可靠收益，本项目测试的是不同的固定卡池和策略基线。 [Ai-vgc 阵容实现](https://raw.githubusercontent.com/Nisanth-Nimashakavi/Ai-vgc/main/src/ai_vgc/nn/preview.py) · [其试验结果](https://github.com/Nisanth-Nimashakavi/Ai-vgc/blob/main/docs/training-ideas.md)

开发集三个对手的成绩为 86/144，Joint 对照为 80/144。原样冻结配置后，`220001` 起的独立 64 个卡池上，依次对 Joint、Tactical 和 Tactical-calibrated 为 72、88、86 胜，各打 128 局。Joint 对照为 64、83、82 胜。平均配对提升 +4.43 个百分点，95% 区间为 +0.78 至 +8.33。直接对 Joint 的 56.25% 胜率的配对区间为 51.56%–60.94%。每个对手的 128 局有 27 次改选，12 次评估回退。Windows 并行测试的预览 P95 为 0.743–0.750 秒，P99 为 1.019–1.046 秒。768 局无非法动作和超时。另在 12 份官方公开预览回放验证合法输出，全部完成且有两次改选。 [首次盲测清单](data/lineup_benchmark.json)

保持原配置，第二批 `230001` 起的 64 个独立卡池换成 Joint、Pivot 和 Pivot-base。Pivot 系列沿用标准启发式预览和原版 Ace Trainer 战斗。双方各 384 局，回放组依次赢 74、82、82 局，Joint 对照为 64、80、79 局。平均配对提升 +3.91 个百分点，95% 区间为 +0.26 至 +7.81。两批对 Joint 合计 146/256，配对胜率为 57.03%，合并区间为 53.91%–60.55%。两批的桌面预览 P95 均低于 0.79 秒，P99 低于 1.05 秒。两次盲测共 1,536 局，无非法动作或超时，墙钟约 175 秒。 [扩展盲测清单](data/lineup_confirm_benchmark.json)

保留阵容回放加 Joint 为最佳离线混合策略。回退覆盖预览未唯一匹配的物种，例如本地公开预览将快速流形态显示为 `urshifu` 的情形。当前代码使用 Python 和同一个 Showdown 依赖，阵容回放没有额外权重。M4 Pro 预览延迟、实际平台与 DeepSeek 联合决策待验证。正式入口继续使用原有 Joint + DeepSeek。

## Draft、Preview、Battle 联合实验

### 端到端目标与采样

原 Draft 规则主要奖励未覆盖的辅助招式和 Protect。假设用完整胜负估计固定卡池的单卡强度及搭配，可在实际战斗前获得更大的胜率收益。采样复用 `play_game`、蛇形选秀、道具条款和合法 `Decision`。随机偏好探索不同阵容，Preview 在 Tactical 首选与多样备选之间采样，Battle 主要使用 Joint，并混入 Tactical。每个池换座。只记录公开卡 ID、两边模拟者各自选出的阵容角色和终局胜负。推理中对手实际带入四只始终未知，六卡待选阵容是公开的。

VGC-Bench 的规模与跨阵容泛化实验支持按池留出，并保留不同风格对手做交叉评估。Ai-vgc 的候选阵容矩阵强调用新对局验证预测胜率。PokéChamp 的接口启发本项目把 LLM 用于提出或选择候选，并由可控模拟确认结果。其单打 minimax 与大规模 PPO 留作方法参考，课程的 Draft 值得优先针对性学习。 [VGC-Bench](https://arxiv.org/html/2506.10326v3#S5) · [VGC-Bench 训练源码](https://raw.githubusercontent.com/cameronangliss/VGC-Bench/main/vgc_bench/train.py) · [Ai-vgc Preview](https://raw.githubusercontent.com/Nisanth-Nimashakavi/Ai-vgc/main/src/ai_vgc/nn/preview.py) · [PokéChamp 论文](https://proceedings.mlr.press/v267/karten25a.html) · [PokéChamp 代码](https://raw.githubusercontent.com/sethkarten/pokechamp/main/pokechamp/llm_player.py)

192 局试跑使用八个进程，约 93.2 局／秒。试点训练和验证分别为 4,096 和 768 局，种子起点为 `250001`、`270001`。扩展训练从 `310001` 开始，在部分采样者中引入冻结试点单卡分并保留随机探索。12 个进程在 76.74 秒内生成 20,480 局，约 266.9 局／秒，无失效换座对。两次拟合使用 PyTorch 2.10.0+cu128、种子 `921`、AdamW、最多 24 epochs，并按同一独立验证集上的两类位置评分选择 checkpoint。扩展模型包含 44,929 个参数，宽度 64，注意力四头。4090 拟合为 6.51 秒，峰值 PyTorch 分配为 82.99 MiB。试点宽度 48，26,017 参数，拟合为 1.57 秒，峰值 77.32 MiB。两套模型保存了权重、数据划分、配置、指标和设备验证样本。

模型使用卡 ID 和角色嵌入、队内共享注意力和座位反对称分数。训练从相同终局学习未知阵容角色的六卡 Draft 价值，以及带 4、首发 2 的 Preview 价值。岭回归拟合的单卡分也是独立对照。神经 Draft 对每个当前合法选卡作 12 次共同随机补全，包含简单启发式和拟合单卡分两个对手倾向，并遵守两边道具条款。模型为补全的六卡队伍打分。最终命令仍来自原始 `Decision.actions`。

### 阶段消融

试点在四个开发对手中，相对原选卡的平均收益为 +16.80 个百分点，95% 配对区间 +11.13 至 +22.27。单卡分为 +16.02。随后在另一开发集 `284001` 的 512 局同设置对照中，原选卡、试点神经、扩展神经、扩展单卡分依次获 281、413、437、438 胜。固定扩展神经 Draft 和相同 Preview，切换己方战斗至冻结的 PPO v2，胜场从 437 降到 379。保留 Joint 战斗。

Preview 消融在 `282001` 开始的 24 个新池独立固定 Draft。每组对 Joint 与 Pivot-base 共 96 局。原阵容搜索 58 胜，神经模型直接改选 55 胜，神经模型为搜索补候选也是 58 胜。模型加入的候选 22 次中均未被搜索实际采用。完整系统保留原阵容搜索。

### DeepSeek 教师离线验证

用 12 份公开阵容回放中存在本地策略分歧的场景，生成合法选项 ID 提示、独立 Showdown 答案数据和评分器。提示仅有公开卡信息和候选，答案保存在单独文件。960 局验证对局覆盖两个对手阵容假设和独立共同随机数。用 Tactical 回复作为免费对照，和本地阵容搜索相比，12 个入选的困难样本平均低 40.10 个百分点。这个数只适用于筛选出的分歧样本，不能当作总胜率。未调用 API，也未估计 DeepSeek 胜率。准备付费比较时可先限制到这 12 次候选选择，按实际计费请求另行批准预算。正式入口继续由 DeepSeek 实质参与选择。

### 冻结全流程对阵和选择

冻结训练产物后，用 `290001` 起的 64 个新池。每组每对手 128 场换座，三组各 640 局。己方三组均用同一个阵容搜索 + Joint，更换 Draft 的是单卡分与神经选卡两组。Pivot 系列的 Draft 与 Preview 采用现有启发式，战斗使用固定 Ace Trainer。旧 PPO 是 `ppo-v2/annealed/step-0004`，基准模式下 Draft 和 Preview 走其既有启发式入口。

| 对手 | 旧最强混合策略 | 扩展单卡分 | 扩展神经 Draft |
| --- | ---: | ---: | ---: |
| 阵容搜索 + Joint | 64 | 113 | 114 |
| Joint | 66 | 116 | 116 |
| Pivot | 85 | 112 | 112 |
| Pivot-base | 82 | 107 | 107 |
| 冻结 PPO v2 | 91 | 113 | 116 |
| **合计** | **388/640** | **561/640** | **565/640** |

单卡分相对旧最强对照增加 27.03 个百分点，95% 配对区间为 +21.56 至 +32.34。神经 Draft 增加 27.66 个百分点，区间 +22.34 至 +32.97。神经相对单卡分的差值为 +0.63 个百分点，区间 −0.63 至 +2.03。另在 `300001` 起的 64 个新池直接交手，神经胜 58/128，95% 配对区间 38.28%–51.56%。选用扩展单卡分作为当前最佳离线配置，神经计划继续保留。两组的 Preview 和 Battle 相同，选用的单卡分 P95/P99 不高于 0.14/0.26 ms，神经选卡为 11.06/12.37 ms。单卡分使用同一个 CPU 可加载 checkpoint，约 180.36 KiB。整套盲测无非法动作、超时和执行错误。 [全流程盲测](data/roster_benchmark.json) · [直接消融](data/roster_mirror_benchmark.json)

公开 Testing 回放的 52 个实际选秀池与训练和验证卡池无交集。选定配置与旧最强对照在本地换座用新 RNG 复核，单卡分赢 84/104，95% 配对区间 70.19%–90.38%。扩展训练用同一数据与种子重跑，权重逐元素相同。CPU 和 CUDA 通过八个便携样本。后续训练应建立两边都使用强选卡的对手池，在难选的前几手做受控改选，检验搭配模型能否在更强对手下超越单卡分。

## 早期原型设计备忘

1. **保持一个公共状态接口**。从 18 张池和已公开的选秀记录匹配固定 set，从 6 只预览及后来已露面成员推断对手带入的 4 只。为每个未确认字段记录来源及未知标记。搜索只接收当前 `Decision`、同局历史和固定卡资料。每个假设世界检查己方合法掩码是否与官方一致。强制换人、Fake Out 刚入场、Choice 锁、Trick Room、天气结束和重定向需要专门的残局样本。检查移除未来对手动作后结果仍可复现，并验证互不相干的分支执行顺序不影响结果。
2. **第一版搜索用一回合行动矩阵**。我方从 Joint 和 Tactical 保留 4–6 个多样化联合候选，始终保留 Joint 首选。对手在每个假设世界生成攻击、Protect、换人和控速等 3–4 个合法联合回应。对各行动对用相同的少量 RNG seed 比较，通过 Showdown 结算到下一共同决策边界。若有强制换人，记录状态或由固定替换策略显式结算。跨假设世界只能用一套己方选择。先用 HP、剩余队员与速度控制构成叶节点评分，比较加权期望及保守分位数。只对完整覆盖的矩阵做改选评估，一次 MCP 响应的墙钟上限需要由 M4 实测决定。
3. **小型实体模型**。首选一到两层、宽度 64、四头自注意力，将双方 12 个队伍实体、己方当前招式与场地编码成上下文。实体用已验证卡 ID、招式、道具和特性的小嵌入，加 HP、状态、角色、已带入／已见和未知标记。由共享的槽位编码器读取行动者、招式和目标实体，联合头额外看集火、覆盖、换人及 Protect 搭配。对 `Decision.actions` 的实际动态列表打分，再只遮蔽批处理 padding。Value 头共享状态主干，在可见信息下估计终局得分，训练目标取胜 1、平 0.5、负 0。目标体量小于 20 万参数，CPU 与 MPS 都走同一套编码，并在 M4 实测单样本 P95/P99 后再选设备。
4. **用搜索结果训练，而后进入自对弈**。若搜索对 Joint 的换座得分有可靠增益，在未用于晋级的训练局中保存搜索产生的各候选相对收益、分支样本数与合法掩码。用这些收益形成软 Policy 目标，并用独立继续对局的实际终局训练和校准 Value。未能可靠恢复或预算打断的决策保留源标签，单独统计搜索覆盖率。之后比较相同 rollout 预算的短 PPO 对照，按局采样 Joint、Pivot、Pivot-base 和冻结的历史模型。先固定混合对手，随后再测 Fictitious Play 或 PFSP。若发生策略漂移，再对照冻结 Search 初始化的 KL 锚定。
5. **以完整对局决定晋级**。迭代用开发对局，使用独立的冻结清单做配对换座复验，并保留新的池和对手做发布确认。记录胜率、配对分差置信区间、非法率、超时、P95/P99、整局时钟和 Search 改选失败。单独测 `Joint + LLM`、`Search + LLM` 与各自本地控制的胜率和实际模型选择比例。在 MacBook 实测 MCP 与 LLM 延迟后选择部署方式。

## 复现命令

从 `pokemon/` 运行。`--ace-trainer` 传入固定的 Ace Trainer `b74574a` checkout。可将 `history-hard` 原样替换为 `shortlist-hard` 或 `history-soft`。在读取 v1 权重时可加 `--bc-shortlist` 做同权重候选消融。

```sh
python -m pokemon_agent bc-train --variant history-hard --output checkpoints/bc-v2/history-hard --train-pairs 160 --test-pairs 40 --test-seed 94001 --ace-trainer /path/to/ace-trainer
python -m pokemon_agent bench --frozen --battle-only --pair-results --left bc --right pivot --bc-checkpoint checkpoints/bc-v2/history-hard --ace-trainer /path/to/ace-trainer
python -m pokemon_agent bc-verify checkpoints/bc-v2/history-hard --device cpu
python -m pokemon_agent restore-audit --games 32
```
