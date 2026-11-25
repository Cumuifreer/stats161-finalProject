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
