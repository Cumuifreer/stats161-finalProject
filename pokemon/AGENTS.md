# STAT 161 Pokémon 工作约定

本文件适用于 `pokemon/`。用户负责 Pokémon VGC Doubles Draft。Git 根目录是上一级，仓库还包含团队的其他内容。开发、测试和文档修改以当前 Pokémon 目录为范围。GitHub 推送由用户执行。

新会话先读 [PROGRESS.md](PROGRESS.md)、[DECISIONS.md](DECISIONS.md) 和 [README.md](README.md)。前两份记录当前验证证据和技术决策，README 提供命令。

## 课程约束

- 正式赛制以本地 [课程规则 PDF](<参考材料/AI Agent Games_ Rules and Self-hosted Play.pdf>) 和课程通知为准。平台主页的旧赛事日期与赛制用于历史背景。技术接口与计时对照[官方 Pokémon API 文档](https://api.altruagent-game.com/skill/pokemon)和实战回放。
- 正式比赛必须让 LLM 根据当前局面实际影响合法动作的选择。默认启发式用于协议验证、数据采集和基准。正式对局使用经过实测的 LLM 入口，核对回放里的 `source`、超时和降级比例。
- 课程玩法是 18 张共享池蛇形选 6、带 4、首发 2。道具冲突限制本方的合法选卡动作。战斗每回合提交双槽联合动作，支持换人和强制替换。平台禁用太晶化。课程循环赛每对一场，前八进入三局两胜淘汰赛。

## 实现与验证

- 保持修改集中、命名准确、代码可读。先用仓库现有组件解决问题，验证受影响的正常入口和边界条件。在确有回归风险时补针对性测试。
- 复用官方 Starter 的认证、席位和 MCP 通讯，以及本地 Showdown 的战斗规则。联机、模拟、搜索和训练共用 `pokemon_agent/game.py` 的 `Decision`、状态字段和合法联合动作，保持玩家可见信息边界。
- 改动协议或模拟器后，运行单元测试、`replay` 和 `parity`。对策略改动使用同池同种子、交换席位的配对 Benchmark，报告胜率、失败、计时和回退。用保留的种子、池子或对手评估泛化。
- `data/historical_cards.json` 的 48 张配置已与本届 Testing 样本逐张核对。变更卡表前用新回放与 `catalog` 查证。训练脚本采用已核对的配置并保留卡表、规则和特征版本。
- 训练目标是对局胜率。先验证 BC 示范、PPO 自对弈和搜索在 Benchmark 中各自的增益。4090 按 CUDA 训练，48 GB M4 Pro MacBook Pro 按单次决策和端到端延迟实测选择推理后端。

## 凭据和本地文件

- 直接让 Starter 从本目录的 `.env` 读取 Official Agent Key。需要付费模型时使用配置好的模型凭据，避免在命令、输出、文档、回放和 Git 中出现凭据值。
- 实战回放留在 `captures/`。课程材料、环境、训练权重、日志、临时结果和四份交接文档使用 `.gitignore` 中的本地规则。保留项目所需的源码、锁文件、测试以及已核对的卡表。
- Testing 由仪表板创建，Starter 的 `--match` 接收已分配的对局。批次开始前确认本目录只有一个采集进程，结束后验证双方 JSONL 并停止该进程。
