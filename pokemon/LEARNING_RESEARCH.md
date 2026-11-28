# Pokémon Agent 的终局学习与 DeepSeek 研究

## 独立盲测结果

锁定的第一代 Draft＋Preview 神经网络和原 Battle 搜索，在 256 个新卡池的完整换座比赛中对 Quicksilver 公开离线版胜 **421/512（82.23%）**。按卡池划分的 95% 配对区间为 **78.71%–85.55%**。实测达到 80% 点目标。先前冻结版独立盲测胜 391/512（76.37%），两批互不重叠的卡池分别重采样，差为 +5.86 个百分点，95% 区间为 +0.78 至 +10.94 个百分点。首选座胜 203/256，后选座胜 218/256。

网络改变了 186 次 Draft 决策，Showdown 采用了 164 次网络提名的 Preview 阵容。Draft、Preview、Battle 决策 P95 分别为 4.19、46.29、4.52 秒。Quicksilver 盲测墙钟 6,472.85 秒，先前冻结版和此候选的版本文件均已复核。[盲测对局](benchmark_results/research-learn80/learned-draft-preview-blind-v1/matches.json) [目标判定](benchmark_results/research-learn80/learned-draft-preview-blind-v1/acceptance.json) [冻结包](benchmark_results/research-learn80/learned-draft-preview-blind-v1/frozen-candidate.zip)

### 同一锁定版的泛化

预登记的 64 池子集，每个对手完整换座 128 场。六组和 Quicksilver 盲测共 1,280 场，审计覆盖 15,360 次选卡、2,560 次阵容选择和 23,252 次 Battle 决策。全部对局无非法动作、超时或时间银行判负。六组泛化墙钟 3,805.64 秒，整个最终验证墙钟约 2 小时 51 分钟，API 支出 $0。[泛化完整记录](benchmark_results/research-learn80/learned-draft-preview-blind-v1/generalization.json) [审计](benchmark_results/research-learn80/learned-draft-preview-blind-v1/audit.json)

| 对手 | 胜场 | 实测胜率 | 95% 配对区间 | 相对前一独立盲测的点差 |
| --- | ---: | ---: | ---: | ---: |
| Ace Trainer 本地版 | 115/128 | 89.84% | 84.38%–94.53% | −1.56 个百分点 |
| Ace Trainer Pivot | 118/128 | 92.19% | 86.72%–96.88% | +2.34 个百分点 |
| Pivot-cycle | 116/128 | 90.62% | 85.16%–95.31% | +4.69 个百分点 |
| Pivot-pressure | 112/128 | 87.50% | 82.03%–92.19% | −3.13 个百分点 |
| 早期冠军 | 119/128 | 92.97% | 88.28%–96.88% | +8.59 个百分点 |
| 70% 冻结版 | 80/128 | 62.50% | 54.69%–70.31% | +3.13 个百分点 |

## 验收基线

前一冻结搜索版在 256 个未参与训练或选择的卡池上对 Quicksilver 公开离线版胜 391/512（76.37%，95% 配对区间 72.66%–79.88%）。预先锁定的其他对手子集为每个对手 128 场，Ace Trainer 本地版胜 117 场，Pivot 胜 115 场，Pivot-cycle 胜 110 场，Pivot-pressure 胜 116 场。公开成绩覆盖完整 Draft、Preview、Battle，使用共同卡池与独立比赛种子换座。对手公开提交为 `198ee85c350ef06e0239afe8ae83563f3035da43`，运行公开搜索配置，私有原冠军和可选 Jev 无法核验。178 个基线文件重新校验通过。[盲测、配置与对手核查](QUICKSILVER80.md)

## Battle 的终局训练

网络参数量为 1,226,595，结合场上实体和候选动作注意力、最近四次己方决策的 GRU，以及模仿、动作胜率和局面胜率三个输出。模型输入限玩家视图及固定公开卡表。后续回合或终局只进入训练标签。数据按完整卡池切分，双方和同一卡池的全部实验都留在同一组。训练样本注册的卡池不进入最终盲测。

| 训练轮 | 新卡池 | 新完整对局 | 双方模型自我对弈 | 新局面 | RTX 4090 优化耗时 | 优化更新 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 策略群初始化 | 3,072 | 6,144 | 0 | 98,949 | 56.19 秒 | 3,072 |
| 模型进入策略群 | 8,192 | 16,384 | 8,738 | 254,422 | 86.85 秒 | 3,200 |
| 同局面动作终局对照 | 640 | 1,280 | 0 | 9,754 | 61.85 秒 | 3,405 |

新数据共有 23,808 场完整比赛和 363,125 个决策局面。第二轮在模型操控的 184,938 个可选决策中，65,833 次模型首选和教师首选不同。探索采样仍可能选择其他动作。第三轮从 1,115 个训练局面生成 26,760 条成对终局续局。合法备选动作共享替代随机抽样，模型的训练输入仍取自分支发生前的玩家视图。第三轮每训练批额外采样 24 个有配对标签的局面。

三轮累计 9,677 次优化，训练循环合计 204.89 秒，4090 峰值张量分配 4.86 GiB。三批数据生成墙钟分别为 122.51、293.15 和 433.59 秒。没有为训练调用付费 API。初始化同时参考历史独立开发池的既有配对分支。第二、三轮恢复上一轮网络及优化器状态，在新状态分布继续训练。

第一轮动作胜率预测 NLL 从 0.698 降至 0.486，之后升至 1.045。保留第 2 轮。第二轮在扩大后的留出集上从 0.499 降至 0.482。第三轮只降约 0.001，同局面动作终局数据当前主要改善实际动作选择的次序。具体任务由下表的完整比赛检验。各阶段的留出集不同，损失只能与本阶段起点直接比较。[训练检查点](checkpoints/research-learn80)

| 独立开发卡池组 | Battle 策略 | 对 Forecast，胜 / 场 | 对 Forecast-replies，胜 / 场 |
| --- | --- | ---: | ---: |
| 128 池 A | Joint | 117 / 256 | 126 / 256 |
| 128 池 A | 初始化模型 | 107 / 256 | 116 / 256 |
| 128 池 A | 自我对弈模型 | 109 / 256 | 114 / 256 |
| 128 池 B | Joint | 129 / 256 | 129 / 256 |
| 128 池 B | 自我对弈模型 | 111 / 256 | 114 / 256 |
| 128 池 B | 加入终局配对监督的模型 | 120 / 256 | 118 / 256 |

每场完成 Draft、Preview、Battle，三者只更换 Battle 策略。第二组第三轮模型追回 9 胜和 4 胜。相对 Joint 仍低 3.52 和 4.30 个百分点，95% 配对差值区间分别为 −8.98 至 +1.56、−10.16 至 +1.56 个百分点。CPU Battle 决策 P95 为 5.14 和 5.36 毫秒，含编码及策略处理。神经网络全面接管 Battle 的收益没有成立。[测试 A](benchmark_results/research-learn80/standalone-v1-v2/matches.json) [测试 B](benchmark_results/research-learn80/standalone-v2-v3/matches.json)

沿用冻结搜索框架时，第一轮 Battle 网络只为 Showdown 原有动作估值提供有上限的终局先验。32 个开发卡池上，完整比赛由搜索对照的 45/64（70.31%）提高到 47/64（73.44%，95% 配对区间 65.62%–81.25%）。配对净差 +3.13 个百分点，95% 差值区间 −10.94 至 +18.75，增益仍待确认。网络处理 511 个 Battle 决策，编码与推理平均 14.47 毫秒，完整 Battle 决策 P95 为 4.71 秒。双方无非法动作或时钟处罚，墙钟 2,017.35 秒。[搜索与终局网络整赛消融](benchmark_results/research-learn80/neural-population-v1-dev32/matches.json)

后续从公开对手和本地强搜索的开发比赛记录中精确回放 190 场，得到 3,789 个有多动作的决策，其中 3,402 个实际动作位于可重复产生的 32 项合法菜单。第三轮网络据此蒸馏强策略。留出池模仿 NLL 从 2.670 降到 2.373，但动作 Top-1 只由 33.28% 上升到 33.60%，终局值误差扩大。128 个新卡池的换座测试中，对 Forecast 为 130/256，对 Forecast-replies 为 126/256，Joint 同池为 136/256 和 128/256。当前强策略回放样本有限，盲测候选继续使用冻结 Battle 搜索。[蒸馏训练](checkpoints/research-learn80/distilled-search-v1/training.json) [蒸馏完整对照](benchmark_results/research-learn80/distilled-search-qualification-v1/matches.json)

另用隐式 Q 学习的思路，从 42,930 条己方决策序列的 335,965 个训练决策传播下一个可控局面的胜率，并按估计优势训练策略头。RTX 4090 完成 6,750 次更新，优化循环 174.81 秒。保留模型的留出动作 NLL 从 0.483 降到 0.467。192 个新卡池的配对整赛差分别为 −0.78 和 −0.26 个百分点，保留策略在 6,554 次可选决策里只改选 88 次。改进了概率估计的校准，整赛结果仍缺乏正增益。[IQL 原论文](https://arxiv.org/abs/2110.06169) [离线时序训练](checkpoints/research-learn80/temporal-policy-v1/training.json) [时序模型完整对照](benchmark_results/research-learn80/temporal-policy-qualification-v1/matches.json)

## Preview 的整场价值模型

4,096 个新训练池按规则选卡后，每组比较四套己方和两套对方的合法阵容。备选方案共享比赛抽样和后续策略。数据生成共产生 65,536 场模拟到终局的 Battle，压成 32,768 个阵容配对标签。训练使用含角色嵌入的跨队列注意力模型，参数量 538,113。调换双方时，模型的预测胜负对数几率反号。数据生成墙钟 407.62 秒，4090 优化 4.65 秒。留出池选用第 6 轮，NLL 0.6065。[生成记录](benchmark_results/research-learn80/preview-outcome-v1/completed.json) [训练记录](checkpoints/research-learn80/preview-outcome-v1/training.json)

在 256 个全新开发池的完整换座比赛中，学习组只替换我方 Preview，双方沿用同一 Draft 和 Battle 策略。两种对手各分到 128 池。Preview 决策 CPU P95 约 0.31 秒。两种对手中分别有 244 次和 248 次改变实际阵容。[完整对照记录](benchmark_results/research-learn80/preview-qualification-v1/matches.json)

| 对手 | Joint Preview | 学习 Preview | 胜率差和 95% 配对差值区间 |
| --- | ---: | ---: | ---: |
| Forecast | 124 / 256（48.44%） | 157 / 256（61.33%） | +12.89〔+5.47，+20.31〕个百分点 |
| Forecast-replies | 127 / 256（49.61%） | 145 / 256（56.64%） | +7.03〔+0.78，+12.89〕个百分点 |

将同一个 Preview 网络用于提名冻结版 Showdown 的合法阵容后，32 个开发卡池的 Quicksilver 完整比赛为 50/64（78.12%，95% 配对区间 70.31%–85.94%）。搜索对照为 45/64（70.31%），配对增益 +7.81 个百分点，95% 差值区间 −4.69 至 +21.88。Showdown 选用了 25 次网络推荐的阵容。Preview 决策 P95 为 45.03 秒。整组墙钟 1,629.75 秒，测试 API 支出 $0。[公开 Quicksilver 配对消融](benchmark_results/research-learn80/preview-productionlike-dev32/matches.json)

对 Ace Trainer 和 Pivot 的 24 池消融使用已有共同开局搜索作为对照版本，搜索内也只添加 Preview 模型。Ace Trainer 从 40/48 提高到 45/48（93.75%，95% 配对区间 87.50%–100.00%），差 +10.42 个百分点，95% 差值区间 +2.08 至 +18.75。Pivot 从 44/48 提高到 46/48（95.83%，89.58%–100.00%），差 +4.17 个百分点，95% 差值区间 0.00 至 +10.42。分别采用网络提名 17 次与 20 次。两种对照的底座与 Quicksilver 的 32 池试验不同。墙钟 1,339.39 秒，API 支出 $0。[多对手 Preview 消融](benchmark_results/research-learn80/preview-transfer-ace-pivot-dev24/matches.json)

同一个模型也可评估尚未选完的阵容。第九顺位及之后，为每个合法备选预测最多三组候选补全阵容，评估各自可带四只后的整场胜率。模型仅在配对预期改善超过门槛时改选。在另 192 个新卡池的换座整赛中，只改变 Draft，就对 Forecast 从 95/192 提到 120/192，对 Forecast-replies 从 90/192 提到 112/192。增幅分别为 +13.02 和 +11.46 个百分点，95% 配对差值区间分别为 +5.21 至 +20.31 和 +5.73 至 +17.71 个百分点。实际改选 348 次，Draft 决策 CPU P95 为 151–155 毫秒。此结果使用 Joint 的 Preview 和 Battle。主要对手必须经过整赛单独验证。[选卡价值迁移](benchmark_results/research-learn80/draft-qualification-v1-expanded/matches.json)

冻结搜索框架同时加上同一网络的 Draft 终局估值与 Preview 提案后，对 Quicksilver 在 32 个开发池上胜 51/64（79.69%，95% 配对区间 71.88%–87.50%）。相对同池搜索对照的差为 +9.38 个百分点，95% 差值区间 −3.13 至 +21.88。相对仅加 Preview 的差为 +1.56 个百分点，区间 −7.81 至 +10.94。网络审查了 99 次后段 Draft，改选 31 次。Preview 采用了 30 次网络提案。Draft、Preview、Battle 决策 P95 分别为 3.71、44.15、4.03 秒，墙钟 1,617.03 秒。此联合版在独立抽取 256 池之前锁定，同一权重与策略用于额外六组泛化对手。[合并候选开发成绩](benchmark_results/research-learn80/learned-draft-and-preview-dev32/matches.json) [盲测锁](benchmark_results/research-learn80/learned-draft-preview-blind-v1/frozen.json)

盲测锁列出 186 项本地和已知对手输入，抽样排除了 65,346 个已登记卡池。GitHub 冻结包包含项目候选源码、模型和抽样登记，共 132 项，SHA-256 为 `a83044deef0d06cc43c947e55529434c3ec4c68f3ab178a7e8effe909ab3023d`。包内的 `SHA256.json` 可逐项复核。保留在研究工作区的原始完整归档共 394 项，SHA-256 为 `bd9c24669d9718d608e3775ce6d199b8f763106f0f08ef03a859601b7b51ea8c`。外部对手和模拟依赖可按公开提交与锁文件另行安装。[项目冻结包](benchmark_results/research-learn80/learned-draft-preview-blind-v1/frozen-candidate.zip) [本地复现](README.md#offline-research-benchmarks)

第二代阵容模型在第一代实际参与双方 Draft 和 Preview 的分布上继续训练。4,096 个全新训练池生成 98,304 场终局模拟和 49,152 个阵容标签，8,192 场使用双方模型各自首选的阵容。模型在 Draft 做了 8,190 次改选。生成墙钟 1,765.82 秒，4090 优化 12.63 秒、1,445 次更新。534 个留出池的 NLL 从 0.623 降到 0.593。

两代模型分别同时接管 Draft 后段和 Preview，在 384 个新卡池的完整换座比赛中继续保持同一 Joint Battle。对 Forecast，第一代为 261/384（67.97%），第二代为 259/384（67.45%），相对第一代差 −0.52 个百分点，95% 配对区间 −6.25 至 +5.21。对 Forecast-replies，第一代为 254/384（66.15%），第二代为 258/384（67.19%），相对差 +1.04 个百分点，区间 −4.69 至 +6.77。Joint 的两组结果是 197/384 和 188/384。两代网络相对 Joint 的增益均超过 16 个百分点，第二代额外增益仍未证实。盲测版本使用第一代的锁定权重。[第二代训练](checkpoints/research-learn80/preview-population-v2/training.json) [两代完整对照](benchmark_results/research-learn80/preview-generations-dev384/matches.json)

额外训练了 347,153 参数的 Draft 终局网络，用公开 Draft 局面的合法选卡分支直接学习后续胜负。3,072 个新卡池各比较 3 个合法选法、每个选法 2 个终局。形成 18,432 场完整训练对局。数据生成墙钟 1,948.97 秒，4090 优化 2.45 秒。450 个留出卡池最佳预测 NLL 为 0.6818。

在另 384 个新卡池的完整对照中，两组都保留第一代阵容网络和 Joint Battle。早期 Draft 网络审查 3,456 次合法选卡、改选 1,768 次。对 Forecast 从 271/384 提到 275/384，净差 +1.04〔−4.95，+7.03〕个百分点。对 Forecast-replies 从 246/384 提到 294/384，净差 +12.50〔+7.03，+18.23〕个百分点。新配置的 Draft 决策 CPU P95 约 0.154 秒。[Draft 训练记录](checkpoints/research-learn80/draft-outcomes-v1/training.json) [换座结果](benchmark_results/research-learn80/draft-outcome-qualification-v1/matches.json)

对 Quicksilver 的 32 池开发对照已从断点完成。加入早期 Draft 网络后胜 52/64（81.25%，95% 配对区间 70.31%–90.62%），比同池盲测配置的开发成绩 51/64 多一胜。增益 +1.56 个百分点，95% 配对差值区间 −9.38 至 +12.50。网络审查 288 次、改选 63 次。独立验证的最强版本采用第一代阵容网络与冻结 Battle 搜索。[早期 Draft 完整开发对照](benchmark_results/research-learn80/early-draft-and-preview-dev32/matches.json)

## DeepSeek Flash

研究运行限定 `deepseek-flash` 和合法选择。战术实验只请求 Showdown 已经二次复核、且同默认动作分差较小的早期动作。每场最多请求一次，名单与动作均取自玩家视图，服务失败或超过局部时限时采用已算好的本地选择。每条日志保留用量、延迟、验证结果和实际是否改选。价格按响应返回的输入缓存状态与输出 token 计，以官网峰时价格复核，未获响应的超时请求没有供应商用量。[DeepSeek 定价](https://api-docs.deepseek.com/quick_start/pricing/)

| 子实验 | 请求 | 合法回复 | 选择变化 | 请求延迟中位数 | 峰时用量估算 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 本地烟测 | 2 | 2 | 0 次战术改选 | 2.453 秒 | $0.00138 |
| 首发预测，推理模式 | 24 | 6 | 0 个预测改选 | 5.646 秒 | $0.02811 |
| 首发预测，简短模式 | 24 | 23 | 3 个预测改选 | 1.342 秒 | $0.01060 |
| Quicksilver 完整换座对照 | 61 | 41 | 13 次战术改选 | 4.792 秒 | $0.05007 |
| Flash 提案、Showdown 决定 | 32 | 32 | 1 次提案被执行 | 1.249 秒 | $0.01280 |

首发预测每组为 12 个换座开发池。推理模式的 18 条无效回复全部用满了 768 个输出 token。简短模式提高了结构化响应率，但两组首发命中率均为 4/24，与各自的本地先验相同。三次预测变化仅用于离线评分。[首发预测](benchmark_results/research-learn80/deepseek-preview-dev12.json) [简短模式](benchmark_results/research-learn80/deepseek-preview-brief-dev12.json)

战术组在 32 个开发卡池中胜 46/64（71.88%，95% 配对区间 60.94%–82.81%）。同池同模型、只关闭 DeepSeek 的组胜 47/64。配对差 −1.56 个百分点，95% 差值区间 −15.63 至 +12.50。61 次请求中有 10 次超时和 10 条无效回复，合法建议实际改变了 13 次执行动作。请求延迟 P95 为 7.40 秒，完整 Battle 决策 P95 为 8.57 秒。无非法行动或比赛时钟罚则。墙钟 2,569.05 秒。[完整战术消融](benchmark_results/research-learn80/neural-and-flash-v1-dev32/matches.json)

第三种应用让 DeepSeek 先提议合法的控速、保护、换人或干扰动作，然后由 Showdown 复核。16 个换座开发卡池上胜 26/32（81.25%，95% 配对区间 65.62%–93.75%），原 Preview 网络加搜索对照胜 24/32。配对差 +6.25 个百分点，区间 −12.50 至 +21.88。服务提出 13 个不同于本地初选的建议，搜索最终执行其中 1 个。完整 Battle 决策 P95 为 4.72 秒。此样本不能区分模型贡献和整赛波动。[Flash 提案消融](benchmark_results/research-learn80/flash-showdown-proposals-dev16/matches.json)

提案组的 32 场配对中，双方 Draft 全部相同，10 场在触发 LLM 前就选出了不同的我方 Preview。这暴露了以墙钟限制模拟搜索时的运行波动。因此两场净胜不能归因于一次被采纳的 LLM 提案。仅引入 Preview 网络的 Quicksilver 试验中，64 场 Draft 全部相同，37 场选了不同的我方阵容，其中也有搜索原候选的重新排序。

五项 DeepSeek 实验共 143 次请求。可计量的响应合计 4,096 个缓存命中输入、139,897 个未命中输入、50,800 个输出 token。按记录时间和官方分时价格估算约 $0.08867，已知用量按峰时计约 $0.10295。10 次超时的响应未返回用量，最终扣费以供应商账单为准。[DeepSeek 定价](https://api-docs.deepseek.com/quick_start/pricing/)

## 研究参考

VGC-Bench 对不同规模队伍和历史策略池分别评估，结果显示训练队伍越广通常越能泛化，但在单一对局条件下会损失专门表现。本轮借鉴其模仿初始化和历史策略对弈，保留我们自己的 Draft 规则和卡表。历史公开 VGC 日志早期试验的预测输入与本地框架存在太晶化和阵容重叠差异，因此本轮终局网络采用本地模拟器数据。PokéChamp 的 LLM 提案与对手建模为两种 DeepSeek 实验提供了思路，论文发布的对局为 Gen 9 OU。实战访谈提出的速度控制、保护和协同开局进入战略询问，动作仍由本任务的合法菜单界定。[VGC-Bench](https://arxiv.org/html/2506.10326v3) [公开 VGC 日志](https://huggingface.co/datasets/cameronangliss/vgc-battle-logs-sv/tree/dc95950) [PokéChamp](https://arxiv.org/abs/2503.04094) [VGC 选手战术访谈](https://www.pokemon.com/us/features/2025-europe-international-championships-vgc-preview-roundtable)

另训 427,505 参数的人类首发网络，使用 2025 Reg I 公开队表和 32,640 次真实首发选择。跨年度 2026 Reg I 留出集为 7,898 次首发，Top-1 为 22.12%，Top-3 为 48.58%。4090 优化 11.37 秒。转移到 64 场本地 Quicksilver 开发样本时，该网络仅精确预测 5 个首发，本地先验命中 13 个。选卡后的阵容分布和可用战术有差异，本实验暂不把真实 VGC 首发概率引入整赛策略。[跨年度和本地迁移结果](checkpoints/research-learn80/human-leads-v1-64bit/training.json)
