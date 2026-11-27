"""Bounded DeepSeek advice over legal actions and player-visible observations."""

from collections import defaultdict
from datetime import datetime, timezone
import argparse
import json
from os import getenv
from pathlib import Path
from time import perf_counter
from getpass import getpass
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class DeepSeek:
    def __init__(self, key, limit):
        self.key = key
        self.client = build_opener(NoRedirect())
        self.limit = limit
        self.records = []

    def advise(self, decision, candidates, default, *, context=None, timeout=8.0, think=False):
        if len(candidates) < 2 or len(self.records) >= self.limit or timeout < .5:
            return default
        if default not in candidates or any(action not in decision.actions for action in candidates):
            raise ValueError("DeepSeek menu must consist of legal actions including the default")
        payload = {"phase": decision.phase, "board": decision.observation, "move_info": decision.move_info,
                   "options": dict(enumerate(candidates)), "analysis": context}
        content = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
        if len(content) > 24000:
            return default
        start = perf_counter()
        now = datetime.now(timezone.utc)
        record = {"utc": now.isoformat(), "phase": decision.phase, "thinking": think, "changed": False}
        self.records.append(record)
        try:
            body = dict(
                model="deepseek-flash", max_tokens=768 if think else 128,
                response_format={"type": "json_object"}, thinking={"type": "enabled" if think else "disabled"},
                **({"reasoning_effort": "low"} if think else {}),
                messages=[{"role": "system", "content": (
                    "Plan to win a Pokémon doubles game without Terastallization. Choose a supplied joint action. "
                    "HP values are fractions of maximum HP, so 1.0 is full health. Only use known information. "
                    "If given local simulation scores, treat them as noisy estimates with larger values better. "
                    "Consider speed control, redirection, Protect, switch tempo, and the endgame with the reserves. "
                    "For a move, target 1/2 is opposing left/right and -1/-2 is allied left/right. "
                    'Return JSON {"id": 0, "reason": "a brief concrete tactical justification"}.'
                )}, {"role": "user", "content": content}],
            )
            request = Request("https://api.deepseek.com/chat/completions", data=json.dumps(body).encode(),
                              headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            with self.client.open(request, timeout=timeout) as incoming:
                response = json.load(incoming)
            record["model"] = response.get("model")
            if usage := response.get("usage"):
                hit = usage.get("prompt_cache_hit_tokens") or 0
                miss = usage.get("prompt_cache_miss_tokens")
                miss = usage["prompt_tokens"] - hit if miss is None else miss
                output = usage["completion_tokens"]
                record.update(cache_hit=hit, cache_miss=miss, completion=output,
                              peak_usd=(hit * .006 + miss * .30 + output * 1.20) / 1e6)
            choice = json.loads(response["choices"][0]["message"]["content"] or "null")
            if not isinstance(choice, dict) or type(choice.get("id")) is not int or not 0 <= choice["id"] < len(candidates):
                record["outcome"] = "invalid_reply"
                return default
            if perf_counter() - start > timeout:
                record["outcome"] = "late_reply"
                return default
            action = candidates[choice["id"]]
            record.update(outcome="accepted", changed=action != default, id=choice["id"],
                          reason=str(choice.get("reason", ""))[:240])
            return action
        except (URLError, TimeoutError) as error:
            record.update(outcome=type(error).__name__, http=error.code if isinstance(error, HTTPError) else None)
            return default
        except (ValueError, IndexError, KeyError, TypeError):
            record["outcome"] = "invalid_reply"
            return default
        finally:
            record["seconds"] = round(perf_counter() - start, 4)


def credential():
    return getenv("DEEPSEEK_API_KEY") or getpass("DeepSeek API key (input hidden): ")


def main():
    from ..draft_leaf import CATALOG
    from ..draft_league import read, save
    from ..game import BATTLE
    from ..joint import JointPolicy
    from ..sim import SEATS, Showdown, play_game
    from ..tactics import load_card_profiles

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    llm = DeepSeek(credential(), limit=2)
    cards = read(CATALOG)
    profiles = load_card_profiles(cards, with_sets=True)

    class Probe(JointPolicy):
        def __call__(self, decision):
            self.observe(decision)
            ranked = self.rank(decision)
            selected = ranked[0][0]
            if decision.phase == BATTLE and decision.observation["turn"] in (1, 3) and len(ranked) >= 3:
                menu = [action for action, _ in ranked[:5]]
                notes = [{"local_score": round(details.get("score", 0.), 3)} for _, details in ranked[:5]]
                selected = llm.advise(decision, menu, selected, context=notes, timeout=16., think=bool(llm.records))
            self.record(decision, selected)
            return selected

    samples = defaultdict(list)
    with Showdown() as engine:
        winner = play_game(engine, cards, {SEATS[0]: Probe(profiles), SEATS[1]: JointPolicy(profiles)}, args.seed, samples)
    result = {"seed": args.seed, "winner": winner, "llm": llm.records,
              "peak_priced_usage_usd": sum(record.get("peak_usd", 0.) for record in llm.records),
              "clock_events": [event for times in samples.values() for _, _, event in times if event]}
    save(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
