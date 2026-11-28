"""Test strategic LLM lineup forecasts against completed development matches."""

import argparse
from hashlib import sha256
from pathlib import Path
from random import Random
from statistics import mean
import torch

from ..combination_preview import signature
from ..draft85 import deployment
from ..draft_last_pick import mean_ci
from ..draft_leaf import CATALOG
from ..draft_league import read, save
from ..lead85 import LeadPrior
from ..sim import SEATS
from ..tactics import tactical_actions
from .deepseek import DeepSeek, credential


def forecast(cards, prior, game, llm, think):
    actor = game["actor"]
    foe = next(seat for seat in SEATS if seat != actor)
    rosters = [[cards[card] for card in game["roster"][seat]] for seat in (foe, actor)]
    decision = deployment(*rosters)
    ranked = tactical_actions(decision)
    choices, weights = prior.scenarios(decision, [], ranked)
    choices = [decision.validate(choice) for choice in choices]
    default = choices[0]
    selected = llm.advise(
        decision, choices, default, think=think, timeout=12. if think else 5.,
        context={"task": "This is Team Preview from the opponent's perspective. Predict which four the acting player "
                 "will bring and which two will lead. Each option has bring and leads arrays. Consider complementary "
                 "speed control, disruption, pivots, safe damage and a reserve win condition against the other six. "
                 "Use only the listed card sets, not assumed standard moves. Give the index and a short reason. "
                 "There is no previous reveal of the selected four.",
                 "local_prior_by_option": [round(value, 3) for value in weights]},
    )
    actual = signature(game["lineups"][foe])
    truth_baseline, truth_model = signature(default), signature(selected)
    return dict(control_leads=truth_baseline[1] == actual[1], flash_leads=truth_model[1] == actual[1],
                control_lineup=truth_baseline == actual, flash_lineup=truth_model == actual,
                menu_contains_leads=any(signature(action)[1] == actual[1] for action in choices),
                menu_contains_lineup=any(signature(action) == actual for action in choices),
                prediction_changed=truth_model != truth_baseline)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--development", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--pairs", type=int, default=12)
    cli.add_argument("--seed", type=int, default=415731)
    cli.add_argument("--prior", type=Path, default=Path("checkpoints/research80/lead-ensemble-v2.json"))
    cli.add_argument("--exclude-study", type=Path)
    cli.add_argument("--brief", action="store_true")
    args = cli.parse_args()
    if args.output.exists() or not 4 <= args.pairs <= 16 or read(args.development.with_name("frozen.json"))["blind"]:
        cli.error("Use 4–16 paired development pools with a fresh output")
    torch.set_num_threads(1)
    cards = read(CATALOG)
    prior = LeadPrior(args.prior, cards)
    index = {card["card_id"]: card for card in cards}
    data = read(args.development)
    previous = set(read(args.exclude_study)["records"]) if args.exclude_study else set()
    selected = Random(args.seed).sample(sorted(set(data["records"]) - previous), args.pairs)
    llm = DeepSeek(credential(), 2 * len(selected))
    pairs = {}
    for root in selected:
        games = data["records"][root]["quicksilver-public"]
        pairs[root] = [forecast(index, prior, game, llm, not args.brief) for game in games]
        print(f"Forecasts evaluated on {len(pairs)}/{len(selected)} pairs", flush=True)
    report = {}
    for key in pairs[selected[0]][0]:
        rates = [mean(row[key] for row in pair) for pair in pairs.values()]
        report[key] = {"rate": mean(rates), "paired_95": mean_ci(rates)}
    for target in ("leads", "lineup"):
        differences = [mean(int(row[f"flash_{target}"]) - int(row[f"control_{target}"]) for row in pair)
                       for pair in pairs.values()]
        report[f"gain_{target}"] = {"difference": mean(differences), "paired_95": mean_ci(differences)}
    save(args.output, dict(report=report, records=pairs, api=llm.records,
                           peak_priced_usage_usd=sum(row.get("peak_usd", 0.) for row in llm.records),
                           pools={root: data["pools"][root] for root in selected},
                           source=sha256(args.development.read_bytes()).hexdigest()))
    print(report, flush=True)


if __name__ == "__main__":
    main()
