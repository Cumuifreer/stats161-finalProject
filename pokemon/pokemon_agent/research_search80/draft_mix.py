"""Check early Draft picks across several public opponent completion policies."""

from collections import Counter
import json
import os
from pathlib import Path
from random import Random
from statistics import fmean
import subprocess
import sys
from time import perf_counter

if __package__:
    import torch
    from ..draft_bundles import DraftBundles
    from ..draft_forecast import public_draft
    from ..game import DRAFT
    from ..sim import fingerprint


class PublicDraftWorlds:
    def __init__(self, checkout):
        self.checkout = checkout.resolve()

    def __enter__(self):
        env = {key: os.environ[key] for key in ("SystemRoot", "TEMP", "TMP") if key in os.environ}
        self.process = subprocess.Popen(
            [sys.executable, "-IB", "-X", "utf8", str(Path(__file__).resolve())], cwd=self.checkout,
            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding="utf-8",
        )
        return self

    def __exit__(self, *_):
        self.process.stdin.close()
        self.process.stdout.close()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def simulate(self, **payload):
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("The public Draft model worker stopped")
        reply = json.loads(line)
        if reply.get("error"):
            raise RuntimeError(reply["error"])
        return reply["worlds"]


if __package__:
    class MixedDraft(DraftBundles):
        def __init__(self, experiment, *, simulator, **settings):
            super().__init__(experiment, **settings)
            self.simulator = simulator

        @torch.inference_mode()
        def __call__(self, decision):
            started = perf_counter()
            pending = self.second is not None
            anchor = super().__call__(decision)
            if (decision.phase != DRAFT or pending or self.second is not None
                    or decision.observation["pick"] not in (1, 2, 4, 6, 8) or len(decision.actions) < 3):
                return anchor
            try:
                return self._consider(decision, anchor, started)
            except (KeyError, RuntimeError, ValueError, OSError):
                self.counts["mixed_draft_fallbacks"] += 1
                return anchor

        def _consider(self, decision, anchor, started):
            example = self.anchor.projection.options(decision, anchor)
            if example is None:
                return anchor
            actions, generic, _, _, _, _, _ = example
            pool, ours, foes, remaining, pick, schedule = public_draft(decision, self.prior.index)
            choices = actions[:5]
            starts = [self.prior.index[action["card_id"]] for action in choices]
            catalog = self.anchor.original_cards
            published = [{**catalog[i], "index": i, "strength": self.prior.strength[i],
                          "support": self.prior.support[i], "guard": self.prior.guard[i]} for i in pool]
            specialist = self.simulator.simulate(
                cards=published, ours=ours, foes=foes, remaining=remaining, actions=starts, pick=pick, schedule=schedule,
            )
            worlds = [old[:4] + other for old, other in zip(generic[:len(choices)], specialist, strict=True)]
            if any(team is None for group in worlds for team in group):
                return anchor
            estimates = self.model(torch.tensor([team for group in worlds for team in group])).sigmoid().tolist()
            chunks = [estimates[i:i + 6] for i in range(0, len(estimates), 6)]
            values = [.56 * fmean(row[:4]) + .44 * fmean(row[4:]) for row in chunks]
            best = max(range(len(values)), key=values.__getitem__)
            self.counts["mixed_draft_reviewed"] += 1
            if best == 0 or values[best] < values[0] + .025 or fmean(chunks[best][:4]) < fmean(chunks[0][:4]) - .035:
                return anchor
            self.counts["mixed_draft_proposed"] += 1
            gains = [after - before for before, after in zip(chunks[0], chunks[best], strict=True)]
            if values[best] >= values[0] + .060 and min(gains) >= -.020:
                self.counts["mixed_draft_changed"] += 1
                return choices[best]
            left = (decision.seconds_left or 15.) - (perf_counter() - started) - .75
            if left < 1.2:
                return anchor
            sample = [0, 2, 4]
            base, alternative = ([worlds[index][i] for i in sample] for index in (0, best))
            rng = Random(fingerprint(["public-draft-portfolio-v1", decision.observation]))
            if self.anchor.hybrid.rollout.compare(base, alternative, rng, left):
                self.counts["mixed_draft_changed"] += 1
                return choices[best]
            return anchor


def completions(draft, request):
    cards = {card["index"]: card for card in request["cards"]}
    order = request["schedule"]
    first = str(order[0])
    start = request["pick"]
    worlds = []
    for action in request["actions"]:
        variants = []
        for flexible in (False, True):
            rosters = [request["ours"] + [action], list(request["foes"])]
            left = [index for index in request["remaining"] if index != action]
            for ply, seat in enumerate(order[start:], start + 1):
                items = {cards[index]["item"] for index in rosters[seat]}
                legal = [index for index in left if cards[index]["item"] not in items]
                if not legal:
                    rosters = None
                    break
                if seat == 1:
                    obs = {
                        "phase": "draft", "available_cards": [cards[index] for index in legal],
                        "rosters": {str(k): [cards[index] for index in roster] for k, roster in enumerate(rosters)},
                        "pick_number": ply, "first_drafter": first,
                    }
                    scored = draft.top_cards(obs, agent_id="1", k=3)
                    second = flexible and len(scored) > 1 and scored[0][1] - scored[1][1] < 1.4
                    chosen = scored[int(second)][2]["index"]
                else:
                    covered = 0
                    for index in rosters[0]:
                        covered |= cards[index]["support"]
                    chosen = max(legal, key=lambda i: cards[i]["strength"] + .25 * (
                        (cards[i]["support"] & ~covered).bit_count() + cards[i]["guard"]
                    ))
                rosters[seat].append(chosen)
                left.remove(chosen)
            variants.append(rosters)
        worlds.append(variants)
    return worlds


def serve():
    protocol, sys.stdout = sys.stdout, sys.stderr
    sys.path.insert(0, str(Path.cwd()))
    from pokemon import draft

    for line in sys.stdin:
        try:
            result = {"worlds": completions(draft, json.loads(line))}
        except Exception as error:
            result = {"error": f"{type(error).__name__}: {error}"}
        print(json.dumps(result), file=protocol, flush=True)


if __name__ == "__main__":
    serve()
