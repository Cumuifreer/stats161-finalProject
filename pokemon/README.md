# STAT 161 Pokémon agent

The official starter handles AltruAgent authentication, seat discovery and MCP.
The live adapter and local Pokémon Showdown bridge share player-view decisions,
legal joint actions and the same policy functions.

## Status

Official Testing has verified all 48 entries in
[`data/historical_cards.json`](data/historical_cards.json) across 49 distinct
18-card draft pools. The samples contained zero new card IDs or set changes.
Three later official self-play matches verified the DeepSeek agent. Replay passed
for 1,551 decisions from 104 seat captures. The simulator matched both seats'
draft, preview and opening battle decisions for all 52 matches.

The [course rules PDF](<参考材料/AI Agent Games_ Rules and Self-hosted Play.pdf>)
governs the tournament format and requires the LLM to influence decisions in
official play. See [PROGRESS.md](PROGRESS.md) for evidence and open work,
[DECISIONS.md](DECISIONS.md) for design choices and [AGENTS.md](AGENTS.md) for
project instructions.

## Course Testing

Use Python 3.11+ and run these from the `pokemon` directory.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[live,llm]'
test -f .env || cp .env.example .env
```

Put the current Official Agent Key in the ignored `.env` file. For DeepSeek V4
Flash set `POKEMON_LLM_PROVIDER=deepseek` and `DEEPSEEK_API_KEY`. The current
DeepSeek model ID is `deepseek-flash`. OpenAI remains available with
`POKEMON_LLM_PROVIDER=openai` and `OPENAI_API_KEY`. Set `POKEMON_LLM_MODEL`
to override the selected provider's default model. Run the checks first, then
create or join a Pokémon Testing match on the AltruAgent dashboard and leave
one starter process running.

```sh
python -m agent --check-tournament --agent pokemon_agent.live
POKEMON_CAPTURE_DIR=captures python -m agent --match --agent pokemon_agent.live
```

Create **Pokémon VGC Doubles Draft** in the dashboard's Testing tab. A self-play
match with both seats set to your self-hosted agent produces a capture for each
seat. The default factory runs the heuristic for protocol and capture checks.
Stop the starter after the assigned matches finish.

For LLM Testing, install Node.js, run `npm install`, and use the LLM factory.
It ranks lineups using matchup coverage, switch options and speed control.
In battle it ranks actions using approximate damage and turn order, and keeps
Protect, switches, and speed control among the up to six candidates when legal.
When available, the battle view includes the player's legal reserves with their
observed HP, status and moves.
It falls back to the top local action after a failed or late response.
Verify actual LLM use through `source=llm` in the captures.

```sh
python -m agent --check-tournament --agent pokemon_agent.live:create_llm_agent
POKEMON_CAPTURE_DIR=captures python -m agent --match --agent pokemon_agent.live:create_llm_agent
```

For a course competition assignment, start the verified LLM factory with
`--tournament`. Gameplay uses the [official Pokémon MCP API](https://api.altruagent-game.com/skill/pokemon).

## Local evaluation

Install Node.js and run `npm install` from this directory. The bridge drives the
Showdown battle engine directly. The Python side runs the snake draft and gives each
policy its own view. Default benchmarks use two games per seed with seats swapped.

```sh
python -m pokemon_agent bench --pairs 10 --left heuristic --right random
python -m pokemon_agent bench --pairs 49 --seed 1 --testing-pools captures/*.jsonl
python -m pokemon_agent bench --pairs 500 --seed 10001 --left heuristic --right random
python -m pokemon_agent bench --pairs 300 --seed 31001 --battle-only --left tactical --right heuristic
python -m pokemon_agent bench --pairs 300 --seed 42101 --preview-only --left tactical --right heuristic
python -m pokemon_agent bench --pairs 5 --seed 28101 --battle-only --left llm --right llm-heuristic
python -m pokemon_agent replay captures/*.jsonl
python -m pokemon_agent catalog captures/*.jsonl
python -m pokemon_agent parity captures/*.jsonl
python -m unittest discover -s tests
```

Use `--battle-only` to run the same draft and preview policy on both sides.
`--preview-only` fixes the draft and tactical battle policies and compares lineups.
`tactical` takes the best local action under the new ranker. `llm` gives the
LLM the new shortlist and `llm-heuristic` gives it the original top six.
`tactical-calibrated` is a local battle variant. It uses the selected card
catalog's Showdown stats and estimates a subset of weather, screen and item
effects. Preview and draft use the same scoring as `tactical`.
`joint` reranks the same battle shortlist against estimated attack, Protect
and switch replies. It uses both players' public preview and observed battle
positions, along with the fixed card profiles. The estimate tracks whether a
slot is knocked out before it can act, whether Protect buys its partner an
attack, and the speed advantage from Tailwind or Trick Room.
`llm-forecast` adds a worst-reply summary for each candidate to the same
provider prompt as `llm`. Candidate IDs, ordering and fallback are identical
for a given decision. Both LLM variants make billable calls.
Benchmark JSON reports paired win and score rates, draws, game throughput,
decision P95 by phase, LLM choices and departures from each shortlist's first
choice, fallbacks, platform-style automatic actions and battle-time forfeits.
If one seat order fails, neither game from that seed
contributes to the paired rates. Add `--pair-results` to include the left
strategy's score by seed and seat order for matched comparisons across runs.
The local clock charges policy time per seat.
It applies the 15-second draft, 85-second preview and 50-second battle decision
limits, plus the 420-second battle bank. Local timing excludes MCP and server
processing. The two LLM policies make billable calls and use the provider and
API key from the environment or the local `.env`.

### Ace Trainer benchmark opponent

The optional `pivot` opponent runs Ace Trainer's published switch-heavy policy
in a local worker. Install the `live` extra for its Python dependencies.
Keep the Ace Trainer checkout outside this project and pin the version used
for the baseline.

```sh
python -m pip install -e '.[live]'
git clone https://github.com/holmjames/ace-trainer-claude.git /path/to/ace-trainer
git -C /path/to/ace-trainer checkout --detach b74574abb0c5cb26f3107a4d0a389c0dc25c2a52
python -m pokemon_agent bench --pairs 52 --seed 49001 --testing-pools captures/*.jsonl --battle-only --left tactical --right pivot --ace-trainer /path/to/ace-trainer
python -m pokemon_agent bench --pairs 300 --seed 52001 --battle-only --left tactical --right pivot --ace-trainer /path/to/ace-trainer
```

Both sides use the benchmark's shared draft and lineup policies. The worker
gets public drafted sets and battle logs, its own Showdown requests, and the
same legal slot options as the simulator. Its reply must match a legal
`Decision` action. Benchmark win-rate intervals resample complete seat pairs.

`--right pivot-base` uses the same agent, parameters and turn sheet with the
style override disabled. The base turn can still include switches and Protect.
Use the same seed range and pool options for both opponents.

```sh
python -m pokemon_agent bench --pairs 300 --seed 52001 --battle-only --pair-results --left tactical --right pivot-base --ace-trainer /path/to/ace-trainer
python -m pokemon_agent bench --pairs 300 --seed 64001 --battle-only --pair-results --left tactical-calibrated --right pivot --ace-trainer /path/to/ace-trainer
python -m pokemon_agent bench --pairs 300 --seed 64001 --battle-only --pair-results --left tactical-calibrated --right tactical
python -m pokemon_agent bench --pairs 300 --seed 76001 --battle-only --pair-results --left joint --right pivot --ace-trainer /path/to/ace-trainer
python -m pokemon_agent bench --pairs 300 --seed 76001 --battle-only --pair-results --left joint --right pivot-base --ace-trainer /path/to/ace-trainer
python -m pokemon_agent bench --pairs 300 --seed 76001 --battle-only --pair-results --left joint --right tactical
python -m pokemon_agent bench --pairs 52 --seed 77001 --testing-pools captures/*.jsonl --battle-only --pair-results --left joint --right pivot --ace-trainer /path/to/ace-trainer
python -m pokemon_agent bench --pairs 6 --seed 79001 --battle-only --pair-results --left llm-forecast --right pivot --ace-trainer /path/to/ace-trainer
```

Use `--left tactical` as the local control for the `joint` commands.
Use `--left llm` and the same provider and model for the billed prompt control.

Capture is opt-in. It records normalized decisions and selected actions, without
the platform's auth data, free-form messages or opponent IDs. Replay checks those
decisions through the current policy interface. It does not reconstruct the
server's RNG or simulate the captured match from the middle.

Pass both seat files for each official Testing self-play to `parity`. It builds
the observed 18-card draft locally, submits the captured draft and lineup
choices, and compares both sides' normalized boards, legal action sets and move
metadata through the opening battle decision. Later battle turns depend on the
unrecorded server RNG. The benchmark models the documented per-seat clock.

`data/historical_cards.json` is the previous event's observed catalog, sourced
from [Ace Trainer](https://github.com/holmjames/ace-trainer-claude/blob/b74574a/data/cards.json).
All 48 sets match the course Testing samples. It is the default catalog for
local simulation. Use `catalog` to compare future captures for additions and
changes. Use `--testing-pools` to cycle through the ordered 18-card pools seen
at the first pick, or `--cards` to evaluate a specified catalog.

| Paired benchmark | Games | Heuristic vs random | Heuristic P95 |
| --- | ---: | ---: | ---: |
| Observed Testing pools, seed 1 | 98 | 93–5 | 0.54 ms |
| Default catalog, seed 10001 | 1,000 | 885–115 | 0.50 ms |

The local bridge follows [Ace Trainer's simulator design](https://github.com/holmjames/ace-trainer-claude/blob/b74574a/sim/bridge.js).
It exposes the unpicked cards to both players, applies the Item Clause to legal
picks, and tracks opponent moves and other reveals from Showdown's public log.

### Battle behavior cloning

Use the existing pinned Showdown dependency with `npm ci`. For CUDA training,
install the platform-specific PyTorch wheel first. The first Windows run used
PyTorch 2.10.0 with CUDA 12.8. `bc-train` uses `cuda` by default and reports
an error if it cannot access CUDA. PyTorch is optional for other agent modes.

```sh
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pokemon_agent bc-train --output checkpoints/bc-v1 --train-pairs 160 --test-pairs 100 --ace-trainer /path/to/ace-trainer
python -m pokemon_agent bench --left bc --right joint --battle-only --bc-checkpoint checkpoints/bc-v1 --seed 92001 --pairs 100
```

Only the Joint player's battle actions become labels. The other seat cycles
through Tactical, Joint, Pivot and Pivot-base when the Ace Trainer checkout is
supplied. Whole matches and both seat orders stay with their seed. The trainer
rejects seed or pool overlap and records the pool fingerprints in the config.
The existing battle-only draft and preview policies are held fixed. The model
scores the full legal joint-action list and training masks batch padding.

To reproduce on the Mac, sync the source and the ignored `checkpoints/bc-v1/`
directory, install native PyTorch and the same Showdown dependency, and verify
the saved logits and actions. `bench --bc-device mps` uses the same encoder and
checkpoint as the CPU benchmark.

```sh
python -m pip install -e '.[bc]'
npm ci
python -m pokemon_agent bc-verify checkpoints/bc-v1 --device cpu
python -m pokemon_agent bc-verify checkpoints/bc-v1 --device mps
```

### Public Showdown search

The `search` benchmark tests a one-turn planner with Joint as the fallback.
Its separate Showdown worker receives the player's current `Decision`, own
available move PP, published rosters, and completed public protocol events.
It rebuilds a hypothesis after all four brought Pokémon on each side have
been revealed. It reconstructs Tailwind, Trick Room, and timed weather from
public events, and falls back to Joint when an active effect cannot be restored.
The rebuilt legal joint-action set must match the player's current one.

```sh
python -m pokemon_agent bench --frozen-suite data/search_v2_benchmark.json --battle-only --left search --right joint --pair-results
python -m pokemon_agent bench --frozen-suite data/search_v2_benchmark.json --battle-only --left joint --right joint --pair-results
```

The planner compares up to four actions and three replies on three common
random streams. Incomplete batches, unsupported effects, and small or
inconsistent improvements leave the Joint action in place. The search section
of the benchmark JSON reports eligibility, coverage, overrides, and P95/P99 for
completed search calls. This local benchmark uses public Showdown protocol
events. The official integration still uses the existing LLM entry point.

### Battle PPO

Use a version 2 BC checkpoint and the Ace Trainer checkout pinned by
`data/ppo_benchmark.json`. Local CPU workers play complete games with both seat
assignments, using the existing draft and preview policies. The actor scores
each legal joint action. The critic predicts the player-seat terminal return.
The training schedule mixes Joint, Pivot, Pivot-base and frozen policy snapshots.

```sh
python -m pokemon_agent ppo-train --bc-init checkpoints/bc-v2/history-hard --ace-trainer /path/to/ace-trainer --output checkpoints/ppo-v1 --until-update 2
python -m pokemon_agent ppo-train --bc-init checkpoints/bc-v2/history-hard --ace-trainer /path/to/ace-trainer --output checkpoints/ppo-v1 --resume checkpoints/ppo-v1/step-0002
python -m pokemon_agent ppo-verify checkpoints/ppo-v1/step-0006 --device cpu
python -m pokemon_agent bench --frozen-suite data/ppo_benchmark.json --battle-only --pair-results --left ppo --right joint --ppo-checkpoint checkpoints/ppo-v1/step-0006
```

The default workflow uses CUDA for six updates and separate development games
after every second update. The final manifest pins independent seeds and draft
pools. To evaluate another opponent, select a listed opponent with `--right`
and pass the pinned checkout for Pivot or Pivot-base. Use `--left joint` and
omit `--ppo-checkpoint` for the corresponding Joint control.

Sync `config.json`, `state_dict.pt`, and `test_samples.json` from the checkpoint
along with the source and pinned Showdown dependency. `ppo-verify` checks
weights, legal actions, and values against the portable samples. It accepts
`--device cpu`, `cuda`, or `mps`. Training resumption also needs `training_state.pt`.

#### PPO with the Tactical shortlist

`--shortlist` uses the same Tactical candidates as the BC initializer. The
action-space choice is saved with the checkpoint and reused by inference and
historical opponents. The legacy full-list training command remains available
with its original settings above.

```sh
python -m pokemon_agent ppo-train --shortlist --bc-init checkpoints/bc-v2/history-hard --ace-trainer /path/to/ace-trainer --output checkpoints/ppo-v2/annealed --seed 140001 --pairs 32 --total-updates 4 --validation-seed 150001 --validation-pairs 24 --final-suite data/ppo_v2_benchmark.json
python -m pokemon_agent bench --frozen-suite data/ppo_v2_benchmark.json --battle-only --pair-results --value-audit --left ppo --right joint --ppo-checkpoint checkpoints/ppo-v2/annealed/step-0004
```

Repeat the training with `--history-end 0.10` and a separate output path for
the constant-history control. Use `--right ppo-history --history-checkpoint
checkpoints/ppo-v1/step-0006` for the frozen model opponent. For the BC action
ablation, use `--bc-shortlist` and `--bc-full` on the same checkpoint, seeds
and opponent. Opening value diagnostics compare each first scored decision
to that seat's signed terminal result.

### Bounded research experiments

The lineup rollout candidate uses the two public six-card rosters and the
fixed catalog. It plays up to 20 hypothetical games in a separate local
Showdown process and picks a legal lineup only when the same alternative
helps across the tested opponents. Battle decisions use Joint. The preview
budget is five seconds, with the existing Tactical lineup as fallback. The
official starter entry continues to use Joint and DeepSeek.

```sh
python -m pokemon_agent bench --frozen-suite data/lineup_benchmark.json --lineup-only --left lineup-rollout --right joint --pair-results
python -m pokemon_agent bench --frozen-suite data/lineup_confirm_benchmark.json --lineup-only --left lineup-rollout --right pivot --ace-trainer /path/to/ace-trainer --pair-results
```

Use `--left joint` for the corresponding controls. Both arms fix the draft.
Pivot and Pivot-base use the standard heuristic preview, followed by their
pinned Ace Trainer battle policies. This experimental mode uses Python and the
same Node dependency as the other local benchmarks.

The position and intervention experiments also remain reproducible from the
following commands. Neither trained configuration is needed to run lineup
rollouts.

```sh
python -m pokemon_agent forecast-fit --ace-trainer /path/to/ace-trainer --output checkpoints/forecast-v1
python -m pokemon_agent bench --frozen-suite data/forecast_benchmark.json --battle-only --left forecast --right joint --forecast-model checkpoints/forecast-v1/config.json --pair-results
python -m pokemon_agent intervention-train --ace-trainer /path/to/ace-trainer --output checkpoints/intervention-v1
python -m pokemon_agent bench --battle-only --left intervention --right joint --intervention-checkpoint checkpoints/intervention-v1 --seed 210001 --pairs 32 --pair-results
```

### Learning draft and preview values

The roster model predicts full-game outcomes from six-card rosters and optional
lineup roles. `--roster-draft value` uses it to score complete hypothetical
drafts. `--roster-draft prior` uses the fitted per-card strength values in the
same checkpoint. Both can use the unchanged lineup rollout and Joint battle
policies. `--pipeline` allows all three phases to run normally.

```sh
python -m pokemon_agent.roster_data --seed 250001 --pairs 2048 --output checkpoints/roster-v1/train.json
python -m pokemon_agent.roster_data --seed 270001 --pairs 384 --output checkpoints/roster-v1/validation.json
python -m pokemon_agent.roster train --train checkpoints/roster-v1/train.json --validation checkpoints/roster-v1/validation.json --output checkpoints/roster-v1/pilot
python -m pokemon_agent.roster_data --seed 310001 --pairs 10240 --workers 12 --behavior-prior checkpoints/roster-v1/pilot --output checkpoints/roster-v1/expanded.json
python -m pokemon_agent.roster train --train checkpoints/roster-v1/expanded.json --validation checkpoints/roster-v1/validation.json --output checkpoints/roster-v1/expanded --width 64
python -m pokemon_agent.roster verify --checkpoint checkpoints/roster-v1/expanded --device cpu
python -m pokemon_agent bench --pipeline --frozen-suite data/roster_benchmark.json --left roster-net --right lineup-rollout --roster-checkpoint checkpoints/roster-v1/expanded --roster-draft prior --roster-preview rollout --pair-results
```

Use `--roster-draft value` on the same frozen suite for the neural planning
control. For the direct comparison, use `data/roster_mirror_benchmark.json`,
`--left roster-net --right roster-net --roster-draft value --roster-opponent-draft prior`.

For deployment validation, copy `config.json`, `probes.json`, and `state_dict.pt`
with the repository and install the same locked Showdown version. CPU is the
default benchmark device. The parity probe accepts `--device mps` on compatible
Mac installations. The experiment leaves the official entry point unchanged.

The offline Preview teacher panel saves only public prompts and separately
scored simulation evidence. The following commands require no API credentials.

```sh
python -m pokemon_agent.preview_teacher build --captures captures --output benchmark_results/preview-teacher-v1
python -m pokemon_agent.preview_teacher score --panel benchmark_results/preview-teacher-v1 --replies benchmark_results/preview-teacher-v1/tactical-replies.json
```

### Strong Draft opponents

`draft_league` compares two legal picks at a shared early Draft position. Both
branches finish with strong selection policies, Tactical Preview and Joint
Battle. Training uses the two terminal results and their difference. Holdout
games switch seats on the same pool and use full lineup search on both sides.
The saved pool fingerprints prevent overlap with training and validation.

```sh
python -m pokemon_agent.draft_league generate --seed 630001 --pairs 4096 --workers 12 --output checkpoints/draft-v2/train.json
python -m pokemon_agent.draft_league generate --seed 640001 --pairs 640 --workers 8 --output checkpoints/draft-v2/validation.json
python -m pokemon_agent.draft_league train --train checkpoints/draft-v2/train.json --validation checkpoints/draft-v2/validation.json --seed 921 --width 64 --epochs 20 --output checkpoints/draft-v2/neural-pilot
python -m pokemon_agent.draft_league bench --checkpoint checkpoints/draft-v2/neural-pilot --mode search --seed 730001 --pairs 128 --workers 10 --output benchmark_results/draft-v2/blind-search.json
python -m pokemon_agent.draft_league bench --checkpoint checkpoints/draft-v2/neural-pilot --mode search --opponent linear --seed 740001 --pairs 128 --workers 10 --output benchmark_results/draft-v2/confirm-search-vs-linear.json
```

`--mode neural` selects by predicted value. `--mode linear` uses fitted card
scores to value completed drafts. `--mode search` asks the neural model to
nominate one alternative and uses bounded Showdown rollouts to decide whether
to take it. `--mode search-linear` changes only the proposal model. Use a new
output path when replaying a recorded experiment.

To add a frozen learned policy to both sides of sampling, pass
`--behavior checkpoints/draft-v2/neural-pilot` to `generate`. Training accepts
multiple `--train` arguments and `--architecture pairwise` for the model that
fits within-team combinations and directional matchups explicitly.

```sh
python -m pokemon_agent.draft_league generate --seed 700001 --pairs 12288 --workers 12 --behavior checkpoints/draft-v2/neural-pilot --output checkpoints/draft-v2/league-train.json
python -m pokemon_agent.draft_league generate --seed 720001 --pairs 1024 --workers 8 --behavior checkpoints/draft-v2/neural-pilot --output checkpoints/draft-v2/league-validation.json
python -m pokemon_agent.draft_league train --train checkpoints/draft-v2/train.json --train checkpoints/draft-v2/league-train.json --validation checkpoints/draft-v2/league-validation.json --seed 921 --epochs 24 --width 32 --architecture pairwise --output checkpoints/draft-v2/league-pairwise
python -m pokemon_agent.draft_league bench --checkpoint checkpoints/draft-v2/league-pairwise --mode neural --seed 750001 --pairs 160 --workers 8 --output benchmark_results/draft-v2/blind-pairwise.json
python -m pokemon_agent.draft_league bench --checkpoint checkpoints/draft-v2/league-pairwise --mode neural --seed 770001 --pairs 128 --workers 6 --output benchmark_results/draft-v2/confirm-pairwise.json
```

### Forecasting strong Draft opponents

`draft_matchups` freezes the old single-card prior, `league-pairwise` and the
Attention + Search weights. Both seats use strategies from this league. Each
recorded pick contains player-relative ownership, public pick order, a style
label for supervised training, and the chosen legal card. At inference the
planner mixes hypothetical styles, completes the remaining snake turns, and
compares a proposed pick with `league-pairwise` under shared Showdown seeds.
Complete holdout games retain the same lineup search and Joint Battle on both
sides. `--method control` substitutes single-card completions for the learned
predictions while preserving the comparison budget.

```sh
python -m pokemon_agent.draft_matchups generate --seed 820001 --pairs 1152 --workers 12 --search-every 11 --output checkpoints/draft-v3/train.json
python -m pokemon_agent.draft_matchups generate --seed 830001 --pairs 272 --workers 8 --search-every 9 --output checkpoints/draft-v3/validation.json
python -m pokemon_agent.draft_matchups train --train checkpoints/draft-v3/train.json --validation checkpoints/draft-v3/validation.json --seed 919 --width 48 --epochs 20 --output checkpoints/draft-v3/pick-attention
python -m pokemon_agent.draft_matchups bench --checkpoint checkpoints/draft-v3/pick-attention --method hybrid --opponent pairwise --seed 850001 --pairs 128 --workers 10 --output benchmark_results/draft-v3/blind-hybrid-pairwise.json
python -m pokemon_agent.draft_matchups bench --checkpoint checkpoints/draft-v3/pick-attention --method hybrid --opponent pairwise --seed 880001 --pairs 128 --workers 8 --output benchmark_results/draft-v3/confirm-hybrid-pairwise.json
python -m pokemon_agent.draft_matchups bench --checkpoint checkpoints/draft-v3/pick-attention --method hybrid --opponent search --seed 860001 --pairs 128 --workers 10 --output benchmark_results/draft-v3/blind-hybrid-search.json
```

Pass `--architecture anchored` to train the model that predicts a deviation
from the single-card prior, then ranks the remaining legal cards. Repeated
runs require fresh output paths.

### Distilling counterfactual Draft outcomes

The audit evaluates every legal pick twice, using Tactical Preview and full
lineup search with matched Showdown seeds. Later collection keeps the full
Preview and assigns labels to a subset of the expanded legal shortlist. The
model predicts terminal win probability and relative value against the frozen
reference pick. `expanded` and `learned` use the same candidate set and review
limit. `reduced` tests one alternative instead of up to two and checks two
future drafts per comparison. All benchmark opponents are the frozen hybrid.

```sh
python -m pokemon_agent.draft_counterfactual --seed 910001 --positions 32 --workers 12 --repeats 8 --picks 1 3 4 5 6 7 8 9 10 11 12 --audit --output checkpoints/draft-v4/audit.json
python -m pokemon_agent.draft_counterfactual --seed 920001 --positions 104 --workers 13 --repeats 6 --picks 1 3 4 5 6 7 8 9 10 11 12 --output checkpoints/draft-v4/train.json
python -m pokemon_agent.draft_counterfactual --seed 930001 --positions 40 --workers 11 --repeats 6 --picks 1 3 4 5 6 7 8 9 10 11 12 --output checkpoints/draft-v4/validation.json
python -m pokemon_agent.draft_distill train --train checkpoints/draft-v4/audit.json --train checkpoints/draft-v4/train.json --validation checkpoints/draft-v4/validation.json --seed 743 --width 48 --epochs 36 --output checkpoints/draft-v4/action-value
python -m pokemon_agent.draft_distill train --train checkpoints/draft-v4/audit.json --train checkpoints/draft-v4/train.json --validation checkpoints/draft-v4/validation.json --seed 743 --width 32 --epochs 36 --loss ranking --output checkpoints/draft-v4/action-ranking
python -m pokemon_agent.draft_distill bench --method frozen --seed 940001 --pairs 8 --workers 4 --output benchmark_results/draft-v4/frozen-mirror.json
python -m pokemon_agent.draft_distill bench --method expanded --seed 950001 --pairs 96 --workers 10 --output benchmark_results/draft-v4/blind-expanded.json
python -m pokemon_agent.draft_distill bench --method learned --checkpoint checkpoints/draft-v4/action-ranking --seed 950001 --pairs 96 --workers 10 --output benchmark_results/draft-v4/blind-learned.json
python -m pokemon_agent.draft_distill bench --method reduced --checkpoint checkpoints/draft-v4/action-ranking --seed 950001 --pairs 96 --workers 10 --output benchmark_results/draft-v4/blind-reduced.json
python -m pokemon_agent.draft_distill bench --method learned --checkpoint checkpoints/draft-v4/action-ranking --seed 960001 --pairs 192 --workers 10 --output benchmark_results/draft-v4/confirm-learned.json
```

The extra `--picks 2` batches fill the missing second-pick labels. Fresh runs
require new output paths.

```sh
python -m pokemon_agent.draft_counterfactual --seed 970001 --positions 20 --workers 9 --repeats 6 --picks 2 --output checkpoints/draft-v4/pick2-train.json
python -m pokemon_agent.draft_counterfactual --seed 975001 --positions 12 --workers 7 --repeats 6 --picks 2 --output checkpoints/draft-v4/pick2-validation.json
python -m pokemon_agent.draft_distill train --train checkpoints/draft-v4/audit.json --train checkpoints/draft-v4/train.json --train checkpoints/draft-v4/pick2-train.json --validation checkpoints/draft-v4/validation.json --validation checkpoints/draft-v4/pick2-validation.json --seed 743 --width 32 --epochs 36 --loss ranking --output checkpoints/draft-v4/action-ranking-covered
python -m pokemon_agent.draft_distill bench --method learned --checkpoint checkpoints/draft-v4/action-ranking-covered --seed 980001 --pairs 128 --workers 12 --output benchmark_results/draft-v4/blind-covered.json
python -m pokemon_agent.draft_distill bench --method learned --checkpoint checkpoints/draft-v4/action-ranking-covered --seed 990001 --pairs 192 --workers 12 --output benchmark_results/draft-v4/confirm-covered.json
```

### Final-pick review and learned proposals

The final-pick bench shares the unchanged prefix across arms and swaps the
policies' seats in every pool. Only the first drafter owns pick 12. `tactical`
and `full` compare the same two cards and use the same paired battle draws.
`tactical-compute` allows more paired battles under the same wall-clock cap.
`value` tests a direct override using the frozen score. `model` uses the learned
score without extra review, and `model-full` uses it to nominate a card for
full Preview review. The final game retains the original Preview and Joint
Battle in all arms. Extra local replays use separate seeds and are excluded
from the full-game benchmark score.

```sh
python -m pokemon_agent.draft_last_pick generate --seed 1010001 --count 48 --workers 12 --draws 16 --tactical --output checkpoints/draft-v5/audit.json
python -m pokemon_agent.draft_last_pick bench --seed 1020001 --count 96 --workers 12 --opponents hybrid search pairwise --plans tactical full tactical-compute --output benchmark_results/draft-v5/search-pilot.json
python -m pokemon_agent.draft_last_pick generate --seed 1030001 --count 112 --workers 13 --draws 16 --exclude checkpoints/draft-v5/audit.json --output checkpoints/draft-v5/train.json
python -m pokemon_agent.draft_last_pick generate --seed 1040001 --count 56 --workers 11 --draws 16 --exclude checkpoints/draft-v5/audit.json checkpoints/draft-v5/train.json --output checkpoints/draft-v5/validation.json
python -m pokemon_agent.draft_distill train --train checkpoints/draft-v5/audit.json --train checkpoints/draft-v5/train.json --validation checkpoints/draft-v5/validation.json --seed 1125 --epochs 36 --width 32 --loss ranking --output checkpoints/draft-v5/last-rank
python -m pokemon_agent.draft_last_pick bench --seed 1050001 --count 120 --workers 12 --opponents hybrid search pairwise linear --plans tactical full tactical-compute value model model-full --checkpoint checkpoints/draft-v5/last-rank --review-budget 6 --local-draws 8 --exclude benchmark_results/draft-v5/search-pilot.json --output benchmark_results/draft-v5/heldout.json
python -m pokemon_agent.draft_last_pick bench --seed 1060001 --count 192 --workers 12 --opponents hybrid --plans tactical full --review-budget 6 --local-draws 8 --exclude checkpoints/draft-v5/audit.json checkpoints/draft-v5/train.json checkpoints/draft-v5/validation.json benchmark_results/draft-v5/search-pilot.json benchmark_results/draft-v5/heldout.json --output benchmark_results/draft-v5/confirm-hybrid.json
```

The pick-11 audit uses `draft_counterfactual --picks 11`. Pass
`--tail-opponents prior hybrid` to cover two policies that can pick different
cards on the last turn.

```sh
python -m pokemon_agent.draft_counterfactual --seed 1070001 --positions 24 --workers 11 --repeats 8 --picks 11 --audit --output checkpoints/draft-v5/pick11-audit.json
python -m pokemon_agent.draft_counterfactual --seed 1071001 --positions 16 --workers 9 --repeats 8 --picks 11 --tail-opponents prior hybrid --audit --output checkpoints/draft-v5/pick11-opponents.json
```

### Late Draft candidate coverage under a fixed budget

The reference in `draft_coverage` is the frozen final-pick champion. Each
method receives six extra review seconds and up to 180 Showdown starts. The
limit counts internal Preview games. Four-card and all-legal search reallocate
Preview depth and use separate discovery and confirmation draws. The learned
four-card method changes the shortlist under the same budget. `learned-two`
uses the model to nominate one challenger while keeping the champion's full
Preview and ten review pairs. Pick 11 projects
up to two opposing replies. The actual paired game runs the original final-pick
champion whenever the candidate owns pick 12.

```sh
python -m pokemon_agent.draft_last_pick generate --seed 1100001 --count 96 --workers 13 --draws 12 --exclude checkpoints/draft-v5/audit.json checkpoints/draft-v5/train.json checkpoints/draft-v5/validation.json --output checkpoints/draft-v6/last-train.json
python -m pokemon_agent.draft_last_pick generate --seed 1110001 --count 48 --workers 11 --draws 12 --exclude checkpoints/draft-v5/audit.json checkpoints/draft-v5/train.json checkpoints/draft-v5/validation.json checkpoints/draft-v6/last-train.json --output checkpoints/draft-v6/last-validation.json
python -m pokemon_agent.draft_distill train --train checkpoints/draft-v5/audit.json --train checkpoints/draft-v5/train.json --train checkpoints/draft-v6/last-train.json --validation checkpoints/draft-v6/last-validation.json --seed 1147 --width 48 --epochs 36 --loss ranking --output checkpoints/draft-v6/last-rank
python -m pokemon_agent.draft_coverage --pick 12 --seed 1090001 --count 16 --workers 6 --opponents champion search pairwise --methods four all direct-prior --output benchmark_results/draft-v6/coverage-smoke.json
python -m pokemon_agent.draft_coverage --pick 12 --seed 1120001 --count 120 --workers 12 --opponents champion champion search pairwise --methods four all learned direct direct-prior --checkpoint checkpoints/draft-v6/last-rank --exclude benchmark_results/draft-v6/coverage-smoke.json --output benchmark_results/draft-v6/last-heldout.json
python -m pokemon_agent.draft_counterfactual --seed 1140001 --positions 64 --workers 13 --repeats 6 --picks 11 --tail-opponents prior hybrid --reference hybrid --all-legal --output checkpoints/draft-v6/eleventh-train.json
python -m pokemon_agent.draft_counterfactual --seed 1150001 --positions 32 --workers 11 --repeats 8 --picks 11 --tail-opponents prior hybrid --reference hybrid --all-legal --output checkpoints/draft-v6/eleventh-validation.json
python -m pokemon_agent.draft_distill train --train checkpoints/draft-v5/pick11-opponents.json --train checkpoints/draft-v6/eleventh-train.json --validation checkpoints/draft-v6/eleventh-validation.json --seed 1147 --width 32 --epochs 36 --loss ranking --output checkpoints/draft-v6/eleventh-rank
python -m pokemon_agent.draft_coverage --pick 11 --seed 1160001 --count 96 --workers 12 --opponents champion champion search pairwise --methods search11 learned11 --checkpoint checkpoints/draft-v6/eleventh-rank --exclude checkpoints/draft-v6/last-train.json checkpoints/draft-v6/last-validation.json benchmark_results/draft-v6/coverage-smoke.json benchmark_results/draft-v6/last-heldout.json --output benchmark_results/draft-v6/eleventh-heldout.json
python -m pokemon_agent.draft_coverage --pick 12 --seed 1130001 --count 128 --workers 12 --opponents champion --methods learned-two --checkpoint checkpoints/draft-v6/last-rank --exclude benchmark_results/draft-v6/coverage-smoke.json benchmark_results/draft-v6/last-heldout.json benchmark_results/draft-v6/eleventh-heldout.json checkpoints/draft-v6/eleventh-train.json checkpoints/draft-v6/eleventh-validation.json --output benchmark_results/draft-v6/learned-two-heldout.json
```

### Fixed-card combination coverage and Preview learning

The `bc` extra provides the training dependencies. The audit ranks every legal
pair and triple by fixed-card matchup features, and counts historical Draft,
bring-four and lead coverage. Training uses paired full-game outcomes and
compares uniform versus coverage-weighted sampling on the same drafted rosters.
Both learned options fit within the frozen five-candidate Preview search.

```sh
python -m pokemon_agent.combination_preview audit --output benchmark_results/combination-v1/coverage.json
python -m pokemon_agent.combination_preview generate --seed 1220001 --count 12 --workers 6 --coverage benchmark_results/combination-v1/coverage.json --output checkpoints/combination-v1/smoke.json
python -m pokemon_agent.combination_preview generate --seed 1230001 --count 288 --workers 13 --coverage benchmark_results/combination-v1/coverage.json --exclude checkpoints/combination-v1/smoke.json --output checkpoints/combination-v1/train.json
python -m pokemon_agent.combination_preview generate --seed 1240001 --count 96 --workers 7 --draws 4 --coverage benchmark_results/combination-v1/coverage.json --exclude checkpoints/combination-v1/smoke.json --output checkpoints/combination-v1/validation.json
python -m pokemon_agent.combination_preview compare-data --train checkpoints/combination-v1/train.json --validation checkpoints/combination-v1/validation.json --coverage benchmark_results/combination-v1/coverage.json --output benchmark_results/combination-v1/data-comparison.json
python -m pokemon_agent.combination_preview train --arm uniform --train checkpoints/combination-v1/train.json --validation checkpoints/combination-v1/validation.json --output checkpoints/combination-v1/uniform
python -m pokemon_agent.combination_preview train --arm stratified --train checkpoints/combination-v1/train.json --validation checkpoints/combination-v1/validation.json --output checkpoints/combination-v1/stratified
python -m pokemon_agent.combination_preview bench --seed 1250001 --count 160 --workers 12 --rivals champion champion search pairwise --models checkpoints/combination-v1/uniform checkpoints/combination-v1/stratified --exclude checkpoints/combination-v1/smoke.json --output benchmark_results/combination-v1/heldout.json
python -m pokemon_agent.combination_preview audit-cores --seed 1260001 --count 168 --workers 12 --coverage benchmark_results/combination-v1/coverage.json --exclude checkpoints/combination-v1/smoke.json checkpoints/combination-v1/train.json checkpoints/combination-v1/validation.json benchmark_results/combination-v1/heldout.json --output benchmark_results/combination-v1/focused-cores.json
python -m pokemon_agent.combination_preview bench --seed 1270001 --count 192 --workers 12 --rivals champion --models checkpoints/combination-v1/uniform --exclude checkpoints/combination-v1/smoke.json benchmark_results/combination-v1/heldout.json benchmark_results/combination-v1/focused-cores.json --output benchmark_results/combination-v1/confirm-uniform.json
```

### Active sampling for the fifth Preview candidate

The frozen champion and original uniform Preview checkpoint are held fixed.
`survey` drafts legal rosters and measures the disagreement between the old
network and Tactical. `label` produces paired terminal outcomes for two
replies. Both training arms get the same number of labels. In benchmarking,
all arms keep the same five-slot Preview search and replay changed lineups on
separate RNG draws.

```sh
python -m pokemon_agent.preview_active survey --seed 1310001 --count 20 --workers 6 --output checkpoints/preview-active-v1/smoke-survey.json
python -m pokemon_agent.preview_active label --survey checkpoints/preview-active-v1/smoke-survey.json --draws 2 --workers 5 --output checkpoints/preview-active-v1/smoke.json
python -m pokemon_agent.preview_active survey --seed 1320001 --count 300 --workers 13 --rivals champion champion search pairwise --exclude checkpoints/preview-active-v1/smoke-survey.json --output checkpoints/preview-active-v1/train-survey.json
python -m pokemon_agent.preview_active label --survey checkpoints/preview-active-v1/train-survey.json --draws 3 --workers 13 --output checkpoints/preview-active-v1/train.json
python -m pokemon_agent.preview_active survey --seed 1330001 --count 84 --workers 6 --rivals champion search pairwise --exclude checkpoints/preview-active-v1/smoke-survey.json --output checkpoints/preview-active-v1/validation-survey.json
python -m pokemon_agent.preview_active label --survey checkpoints/preview-active-v1/validation-survey.json --draws 5 --validation --workers 7 --output checkpoints/preview-active-v1/validation.json
python -m pokemon_agent.preview_active train --arm uniform --train checkpoints/preview-active-v1/train.json --validation checkpoints/preview-active-v1/validation.json --output checkpoints/preview-active-v1/rank-uniform
python -m pokemon_agent.preview_active train --arm active --train checkpoints/preview-active-v1/train.json --validation checkpoints/preview-active-v1/validation.json --output checkpoints/preview-active-v1/rank-active
python -m pokemon_agent.preview_active bench --seed 1340001 --count 8 --workers 4 --replays 2 --exclude checkpoints/preview-active-v1/smoke-survey.json --output benchmark_results/preview-active-v1/smoke.json
python -m pokemon_agent.preview_active bench --seed 1350001 --count 144 --workers 12 --rivals champion champion search pairwise --models checkpoints/preview-active-v1/rank-uniform checkpoints/preview-active-v1/rank-active --exclude checkpoints/preview-active-v1/smoke-survey.json benchmark_results/preview-active-v1/smoke.json --output benchmark_results/preview-active-v1/heldout.json
python -m pokemon_agent.preview_active bench --seed 1360001 --count 160 --workers 12 --rivals champion --models checkpoints/preview-active-v1/rank-uniform checkpoints/preview-active-v1/rank-active --exclude checkpoints/preview-active-v1/smoke-survey.json benchmark_results/preview-active-v1/smoke.json benchmark_results/preview-active-v1/heldout.json --output benchmark_results/preview-active-v1/confirm.json
```

The follow-up Draft pilot picks alternatives using the first four seeds per
reply and reports matched gains using the remaining four seeds per reply.

```sh
python -m pokemon_agent.draft_counterfactual --seed 1370001 --positions 32 --workers 10 --repeats 8 --picks 1 3 5 7 --tail-opponents pairwise hybrid --reference hybrid --output benchmark_results/preview-active-v1/draft-direction.json
```

### Mid-Draft value against a mixed league

The audit selects on separate battle seeds and reserves a third continuation
style. Training covers picks 1–10. The final Benchmark intervenes once per
seat in each of picks 5–7 and 8–10. Pure and learned search use the same two
projected worlds, Preview depth, terminal pairs and time limits. Actual games
keep the frozen champion's late Draft, complete Preview and Joint Battle.

```sh
python -m pokemon_agent.draft_early generate --seed 1420001 --count 60 --workers 12 --audit --draws 8 --champion-every 5 --exclude checkpoints/draft-early-v1/smoke.json --output checkpoints/draft-early-v1/audit.json
python -m pokemon_agent.draft_early generate --seed 1440001 --count 180 --workers 13 --draws 6 --candidates 6 --champion-every 8 --mix-own --exclude checkpoints/draft-early-v1/smoke.json checkpoints/draft-early-v1/audit.json benchmark_results/draft-early-v1/smoke.json --output checkpoints/draft-early-v1/train.json
python -m pokemon_agent.draft_early generate --seed 1450001 --count 60 --workers 6 --draws 8 --candidates 6 --champion-every 6 --mix-own --exclude checkpoints/draft-early-v1/smoke.json checkpoints/draft-early-v1/audit.json benchmark_results/draft-early-v1/smoke.json --output checkpoints/draft-early-v1/validation.json
python -m pokemon_agent.draft_distill train --train checkpoints/draft-early-v1/train.json --validation checkpoints/draft-early-v1/validation.json --width 64 --epochs 36 --seed 1207 --loss ranking --selection-picks 5 6 7 8 9 10 --output checkpoints/draft-early-v1/rank
python -m pokemon_agent.draft_early bench --seed 1460001 --count 120 --workers 12 --opponents champion champion search pairwise --checkpoint checkpoints/draft-early-v1/rank --methods search learned direct --exclude checkpoints/draft-early-v1/smoke.json checkpoints/draft-early-v1/audit.json benchmark_results/draft-early-v1/smoke.json --output benchmark_results/draft-early-v1/heldout.json
python -m pokemon_agent.draft_early bench --seed 1470001 --count 120 --workers 12 --opponents champion --checkpoint checkpoints/draft-early-v1/rank --methods search learned direct --exclude checkpoints/draft-early-v1/smoke.json checkpoints/draft-early-v1/audit.json benchmark_results/draft-early-v1/smoke.json benchmark_results/draft-early-v1/heldout.json --output benchmark_results/draft-early-v1/confirm.json
```

### Legal continuation replay and terminal value calibration

The replay arms keep the original review limit and proposals. The matched
control adds paired battle seeds to the old projections until it reaches the
measured replay allowance. The two direct leaf arms share the same proposal,
legal continuation policies and acceptance threshold. The guard arm first runs
the complete replay review before applying the learned veto.

```powershell
$cp="checkpoints/draft-early-v1/rank"
$previous=Get-ChildItem benchmark_results/draft-early-v1,checkpoints/draft-early-v1 -File -Filter "*.json" | ForEach-Object FullName
python -m pokemon_agent.draft_early bench --seed 1510001 --count 6 --workers 4 --opponents champion search pairwise --checkpoint $cp --methods search learned replay-search replay-learned --exclude @previous --output benchmark_results/draft-replay-v1/smoke.json
python -m pokemon_agent.draft_early bench --seed 1520001 --count 80 --workers 12 --opponents champion champion search pairwise --checkpoint $cp --methods search learned replay-search replay-learned --exclude @previous benchmark_results/draft-replay-v1/smoke.json --output benchmark_results/draft-replay-v1/heldout.json
python -m pokemon_agent.draft_leaf generate --seed 1530001 --count 18 --workers 4 --draws 4 --output checkpoints/draft-leaf-v1/smoke.json
python -m pokemon_agent.draft_leaf generate --seed 1540001 --count 144 --workers 12 --draws 6 --exclude checkpoints/draft-leaf-v1/smoke.json --output checkpoints/draft-leaf-v1/train.json
python -m pokemon_agent.draft_leaf generate --seed 1550001 --count 48 --workers 7 --draws 8 --exclude checkpoints/draft-leaf-v1/smoke.json --output checkpoints/draft-leaf-v1/validation.json
python -m pokemon_agent.draft_leaf train --train checkpoints/draft-leaf-v1/train.json --validation checkpoints/draft-leaf-v1/validation.json --seed 1607 --output checkpoints/draft-leaf-v1/value
$prior=Get-ChildItem benchmark_results/draft-early-v1,checkpoints/draft-early-v1,benchmark_results/draft-replay-v1,checkpoints/draft-leaf-v1 -File -Filter "*.json" | ForEach-Object FullName
python -m pokemon_agent.draft_early bench --seed 1560001 --count 100 --workers 12 --opponents champion champion champion search pairwise --checkpoint $cp --leaf-checkpoint checkpoints/draft-leaf-v1/value --methods search-matched replay-search replay-learned leaf-control leaf-direct leaf-guard --exclude @prior --output benchmark_results/draft-replay-v1/confirm.json
```

### First-turn Battle counterfactuals with frozen lineups

```powershell
$leaf=Get-ChildItem checkpoints/draft-leaf-v1 -File -Filter "*.json" | ForEach-Object FullName
python -m pokemon_agent.opening_audit --seed 1570001 --count 36 --workers 10 --draws 8 --exclude @leaf --output benchmark_results/opening-audit-v1/pilot.json
python -m pokemon_agent.opening_audit --seed 1580001 --count 84 --workers 12 --draws 8 --exclude @leaf benchmark_results/opening-audit-v1/pilot.json --output benchmark_results/opening-audit-v1/replication.json
python -m pokemon_agent.opening_value train --seed 1609 --workers 12 --train benchmark_results/opening-audit-v1/replication.json --validation benchmark_results/opening-audit-v1/pilot.json --output checkpoints/opening-value-v1/rank
python -m pokemon_agent.opening_value bench --seed 1590001 --count 80 --workers 12 --checkpoint checkpoints/opening-value-v1/rank --exclude @leaf --output benchmark_results/opening-audit-v1/heldout.json
python -m pokemon_agent.opening_value bench --seed 1600001 --count 100 --workers 12 --checkpoint checkpoints/opening-value-v1/rank --exclude @leaf benchmark_results/opening-audit-v1/heldout.json --output benchmark_results/opening-audit-v1/confirm.json
```

### Active sampling for Battle openings

Run from `pokemon/`. The collection step screens openings using the frozen `opening-value-v1/rank` checkpoint and saves the selected rows, legal actions and paired outcomes. Training fine-tunes two copies of that checkpoint using equal numbers of labels. The benchmark runs both checkpoints and the frozen model against the Joint control at the same Draft, Preview, card pool and Showdown seed per seat. The CLI requires fresh output paths.

```powershell
.\.venv\Scripts\python.exe -m pokemon_agent.opening_active collect --seed 1610001 --count 192 --sample-count 72 --validation-seed 1620001 --validation-count 48 --workers 12 --draws 6 --output benchmark_results/opening-active-v1/labels.json
.\.venv\Scripts\python.exe -m pokemon_agent.opening_active train --seed 1637 --device cuda --data benchmark_results/opening-active-v1/labels.json --output checkpoints/opening-active-v1
.\.venv\Scripts\python.exe -m pokemon_agent.opening_active bench --seed 1650001 --count 96 --workers 12 --checkpoint checkpoints/opening-active-v1 --output benchmark_results/opening-active-v1/heldout.json
.\.venv\Scripts\python.exe -m pokemon_agent.opening_active bench --seed 1660001 --count 96 --workers 12 --checkpoint checkpoints/opening-active-v1 --exclude benchmark_results/opening-active-v1/heldout.json --output benchmark_results/opening-active-v1/confirm.json
```

### Full-battle policy and value learning

Run from `pokemon/`. The offline labeling bridge keeps its snapshots in the Showdown worker. The trained agents score the same legal shortlist on the ordinary benchmark bridge. All output paths must be fresh.

```powershell
$data = 'benchmark_results/battle-league-v1'
$checkpoints = 'checkpoints/battle-league-v1'
.\.venv\Scripts\python.exe -m pokemon_agent.battle_value collect --seed 1670001 --count 60 --workers 12 --draws 6 --output benchmark_results/battle-value-v1/audit.json
.\.venv\Scripts\python.exe -m pokemon_agent.battle_value collect --seed 1750001 --count 48 --workers 6 --draws 6 --wide-candidates --exclude benchmark_results/battle-value-v1/audit.json --output benchmark_results/battle-value-v1/coverage.json
.\.venv\Scripts\python.exe -m pokemon_agent.battle_league collect --seed 1680001 --count 480 --workers 12 --draws 2 --output "$data/fit-a.npz"
.\.venv\Scripts\python.exe -m pokemon_agent.battle_league collect --seed 1690001 --count 96 --workers 12 --draws 4 --exclude "$data/fit-a.npz" --output "$data/dev.npz"
.\.venv\Scripts\python.exe -m pokemon_agent.battle_league collect --seed 1730001 --count 72 --workers 6 --draws 4 --late-labels --exclude "$data/fit-a.npz" "$data/dev.npz" --output "$data/dev-late.npz"
foreach ($size in @('small','large')) {
    .\.venv\Scripts\python.exe -m pokemon_agent.battle_league train --seed 1733 --epochs 32 --size $size --data "$data/fit-a.npz" --validation "$data/dev.npz" --output "$checkpoints/$size-a"
}
.\.venv\Scripts\python.exe -m pokemon_agent.battle_league collect --seed 1700001 --count 720 --workers 14 --draws 2 --behavior "$checkpoints/large-a" --exclude "$data/fit-a.npz" "$data/dev.npz" "$data/dev-late.npz" --output "$data/fit-b.npz"
foreach ($name in @('large-a-repeat','history-a','large-b','history-b')) {
    $inputData = @("$data/fit-a.npz")
    if ($name.EndsWith('-b')) { $inputData += "$data/fit-b.npz" }
    $history = @()
    if ($name.StartsWith('history')) { $history = @('--history') }
    .\.venv\Scripts\python.exe -m pokemon_agent.battle_league train --seed 1777 --epochs 24 --size large --imitation-weight .25 --init "$checkpoints/large-a" --data @inputData --validation "$data/dev.npz" "$data/dev-late.npz" --output "$checkpoints/$name" @history
}
.\.venv\Scripts\python.exe -m pokemon_agent.battle_league bench --seed 1710001 --count 60 --workers 6 --model "small-a=$checkpoints/small-a" "large-a=$checkpoints/large-a" --output "$data/development.json"
$models = @("small-a=$checkpoints/small-a","large-a=$checkpoints/large-a","large-a-repeat=$checkpoints/large-a-repeat","history-a=$checkpoints/history-a","large-b=$checkpoints/large-b","history-b=$checkpoints/history-b","value-b=$checkpoints/large-b")
.\.venv\Scripts\python.exe -m pokemon_agent.battle_league bench --seed 1720001 --count 96 --workers 14 --model @models --value-only value-b --exclude "$data/development.json" --output "$data/heldout.json"
.\.venv\Scripts\python.exe -m pokemon_agent.battle_league bench --seed 1740001 --count 96 --workers 14 --model @models --value-only value-b --exclude "$data/development.json" "$data/heldout.json" --output "$data/confirm.json"
```

### Frozen champion strength benchmark

Use a directory containing the [pinned Ace Trainer source](https://github.com/holmjames/ace-trainer-claude/tree/b74574abb0c5cb26f3107a4d0a389c0dc25c2a52) for `$ace`. The runner uses that project's own Draft, Preview and Battle decisions in both external matchups. The code-only entrant runs with `provider=None`, and Pivot uses its upstream local style policy. Each one uses its native 20-second Preview simulation budget. The subprocess receives no API credentials. All paths for results must be fresh.

```powershell
$results = 'benchmark_results/strength-v1'
$ace = '.venv/strength-ace-b74574a'
.\.venv\Scripts\python.exe -m pokemon_agent.strength --opponents random --seed 1800001 --pairs 144 --workers 14 --output "$results/random-a.json"
.\.venv\Scripts\python.exe -m pokemon_agent.strength --opponents random --seed 1810001 --pairs 144 --workers 12 --output "$results/random-b.json"
.\.venv\Scripts\python.exe -m pokemon_agent.strength --opponents heuristic tactical --seed 1820001 --pairs 128 --workers 10 --output "$results/rules.json"
.\.venv\Scripts\python.exe -m pokemon_agent.strength --opponents ace-code pivot --ace-trainer $ace --seed 1830001 --pairs 96 --workers 16 --output "$results/ace.json"
.\.venv\Scripts\python.exe -m pokemon_agent.strength --isolate "$results/ace.json" --workers 12 --output "$results/ace-phases.json"
```

The freeze check verifies learned Draft weight hashes. Every primary matchup runs six snake-draft picks per seat, one Preview per seat, and a Showdown terminal outcome. Both games in a pair share the pool, first draft seat and game RNG seed. Results include per-phase timings, paired confidence intervals and action traces for all losses. `--isolate` replays saved rosters and battle seeds for phase attribution.

### Public-state planning research

The local planning benchmark can combine exact Showdown transitions with inferred opposing reserves, a learned reply distribution and independent forecasts from the published Ace Trainer policies. Each forecast runs on a simulated player's request built from public evidence. The frozen Draft and Preview are the default for these runs. Use a new output path when repeating any command.

```powershell
$planningResults = 'benchmark_results/public-planning-v1'
$planningModels = 'checkpoints/public-planning-v1'
$planningAce = '.venv/strength-ace-b74574a'
.\.venv\Scripts\python.exe -m pokemon_agent.reply_learning --source benchmark_results/strength-v1/ace.json --ace-trainer $planningAce --output "$planningModels/reply"
.\.venv\Scripts\python.exe -m pokemon_agent.planning_bench --opponents ace-code pivot --ace-trainer $planningAce --seed 1900001 --pairs 32 --workers 12 --depth 1 --proxy-replies --reply-model "$planningModels/reply" --compare "$planningResults/control.json" --output "$planningResults/proxy-full-dev.json"
```

Use `planning_replay` for faster Battle comparisons on recorded deployments. Those results are development diagnostics. For acceptance, `planning_blind` locks the candidate and model fingerprints before sampling an independent pool batch.

The frozen continuation planner won 385/512 games against Ace Trainer's local code, 359/512 against Pivot and 409/512 against the frozen champion. Pivot's 95% paired confidence interval was 66.21%–74.02%. The [blind-test summary](benchmark_results/planning-blind-v2/acceptance.json) and [frozen configuration](benchmark_results/planning-blind-v2/frozen.json) record the final result.

To evaluate the same candidate on a fresh blind sample, use the pinned Ace Trainer revision above and check it out with LF line endings. On Windows, set `core.autocrlf=false` for the Ace Trainer clone before checkout.

```powershell
.\.venv\Scripts\python.exe -m pokemon_agent.planning_blind --development "$planningResults/v3-continuation-fresh-dev.json" --ace-trainer $planningAce --reply-model "$planningModels/reply" --pairs 256 --workers 14 --output benchmark_results/planning-blind-reproduction
```

### Offline research benchmarks

The locked learned Draft and Preview candidate won 421 of 512 full matches against the public Quicksilver policy (82.23%, 95% paired interval 78.71%–85.55%). The [learning report](LEARNING_RESEARCH.md) includes the training runs, DeepSeek experiments, full match results, and candidate archive. The [Quicksilver report](QUICKSILVER80.md) describes the pinned public opponent and the earlier 391/512 result. The [85% research report](RESEARCH85.md) records the earlier paired tests against Ace Trainer and Pivot.

Run these commands from `pokemon/` in PowerShell. Install Node.js and Python, and reuse `.venv` if it already exists. Use the pinned fixtures below for repeatable tests. The Ace adapter imports the pinned source directly and uses its Python dependencies. Both opponents run locally without API credentials.

```powershell
python -m venv .venv
$python = './.venv/Scripts/python.exe'
& $python -m pip install -e '.[bc]'
npm ci

git -c core.autocrlf=false clone https://github.com/holmjames/ace-trainer-claude.git .venv/strength-ace-b74574a
git -C .venv/strength-ace-b74574a checkout --detach b74574abb0c5cb26f3107a4d0a389c0dc25c2a52
& $python -m pip install 'httpx==0.28.1' 'python-dotenv>=1.0' 'anthropic==1.12.0' 'mcp>=1.12.0,<2.0.0'
$ace = (Resolve-Path '.venv/strength-ace-b74574a').Path

git clone https://github.com/abubakarsiddik31/quicksilver-hok.git .venv/quicksilver-hok-public
git -C .venv/quicksilver-hok-public checkout --detach 198ee85c350ef06e0239afe8ae83563f3035da43
npm --prefix .venv/quicksilver-hok-public/pokemon/sim ci
$quicksilver = (Resolve-Path '.venv/quicksilver-hok-public').Path

git worktree add --detach .venv/frozen70-checkout 33c5a3c9e634dff86d02476f188178551ab45542
npm --prefix .venv/frozen70-checkout/pokemon ci
$env:POKEMON_FROZEN_BASELINE_PACKAGE = (Resolve-Path '.venv/frozen70-checkout/pokemon/pokemon_agent').Path
```

The next command draws a new blind sample after locking the candidate. Choose an unused output directory. Use `--prior-workspace` for each additional workspace with locally retained pool registries. To resume a run, use the original checkout, fixtures, and output directory and add `--resume`.

```powershell
& $python -B -m pokemon_agent.research_learn80.draft_matches `
  --lineup-net checkpoints/research-learn80/preview-outcome-v1 `
  --ace-trainer $ace --quicksilver $quicksilver `
  --candidate checkpoints/research80/draft-bundles-candidate.json `
  --pairs 256 --workers 12 `
  --generalization ace-code pivot pivot-cycle pivot-pressure champion baseline70 `
  --generalization-pairs 64 --output benchmark_results/learned-draft-preview-reproduction
```
