"""Run a pinned matchup model using only the rosters in the worker request."""

import json
from pathlib import Path
import sys


def run():
    output = sys.stdout
    sys.stdout = sys.stderr
    sys.path.insert(0, str(Path.cwd()))
    from agent.pokemon.lineup import shortlist

    for line in sys.stdin:
        try:
            query = json.loads(line)
            candidates = shortlist(query["own"], query["foe"], list(query["own"]),
                                   top=query["top"], sim_budget_s=query["budget"])
            result = {"candidates": [{"bring": row.bring, "leads": row.leads} for row in candidates]}
        except Exception as error:
            result = {"error": f"{type(error).__name__}: {error}"}
        print(json.dumps(result), file=output, flush=True)


if __name__ == "__main__":
    run()
