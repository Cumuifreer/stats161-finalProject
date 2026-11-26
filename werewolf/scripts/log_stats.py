"""從 logs/ 算報告要用的數字：每局的角色、投票次數、投中狼的比例。

用法：python -m scripts.log_stats   （在 werewolf/ 裡面跑）
"""

import json
from collections import Counter

from agent.agent import LOG_DIR


def main():
    total = Counter()
    for f in sorted(LOG_DIR.glob("*.jsonl")):
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        final = rows[-1]["notes"] or {}
        dead = {int(k): v["role"] for k, v in (final.get("dead") or {}).items()}
        role = final.get("my_role")
        votes = [r for r in rows if r["kind"] == "action" and r["sub_phase"] == "day_vote"]
        on_wolf = sum(1 for r in votes if str(r["choice"]).isdigit() and dead.get(int(r["choice"])) == "wolf")
        abstain = sum(1 for r in votes if r["choice"] == "7")
        print(f"{f.stem}: role={role} votes={len(votes)} hit_wolf={on_wolf} abstain={abstain} "
              f"wolves_dead={sum(1 for v in dead.values() if v == 'wolf')}")
        total.update(role=0, **{f"games_{role}": 1}, votes=len(votes), hit_wolf=on_wolf, abstain=abstain)
    if total["votes"]:
        print(f"\n共 {sum(v for k, v in total.items() if k.startswith('games_'))} 個座位，"
              f"投票 {total['votes']} 次，投中狼 {total['hit_wolf'] / total['votes']:.0%}，棄權 {total['abstain']} 次")
    # 註：只算得到「後來死掉、身份公開」的狼；活到最後的狼這裡看不到


if __name__ == "__main__":
    main()
