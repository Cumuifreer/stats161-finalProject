"""賽後檢討 (Chain-of-Hindsight)：讀一局的 log，請 LLM 寫 3 條教訓存進 agent/lessons.md。
下一局 agent 會自動把 lessons.md 放進 prompt。

用法：python -m scripts.reflect                     # 讀最新一局（在 werewolf/ 裡面跑）
      python -m scripts.reflect logs/xxx.jsonl      # 指定某一局
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent.agent import LESSONS_FILE, LOG_DIR, pick_provider
from examples.llm.base import object_schema

KEEP = 15  # lessons.md 最多留幾條，太多 prompt 會太長


def main():
    load_dotenv()
    if len(sys.argv) > 1:
        files = [Path(sys.argv[1])]
    else:
        files = sorted(LOG_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)[-1:]
    if not files:
        sys.exit("logs/ 裡面沒有紀錄，先打一場。")
    rows = [json.loads(l) for l in files[0].read_text(encoding="utf-8").splitlines() if l.strip()]
    last = rows[-1]["notes"] or {}

    prompt = {
        "task": "You played one Werewolf game. Review your decisions against what was revealed later "
                "(dead players' true roles). Write up to 3 short, concrete lessons for future games "
                "(under 25 words each). Focus on mistakes you can avoid next time.",
        "final_notebook": last,
        "your_decisions": [{k: r.get(k) for k in ("kind", "sub_phase", "choice", "extra")} for r in rows],
    }
    schema = object_schema({"lessons": {"type": "array", "items": {"type": "string"}}}, reasoning=False)
    answer = pick_provider().complete_structured(
        [{"role": "system", "content": "You are a Werewolf coach."},
         {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, default=str)}],
        "lessons", schema)

    old = [l for l in LESSONS_FILE.read_text(encoding="utf-8").splitlines() if l.startswith("- ")] \
        if LESSONS_FILE.exists() else []
    new = [f"- {x.strip()}" for x in answer.get("lessons", [])[:3] if x.strip()]
    LESSONS_FILE.write_text("\n".join((old + new)[-KEEP:]) + "\n", encoding="utf-8")
    print("新增的教訓：", *new, sep="\n")


if __name__ == "__main__":
    main()
