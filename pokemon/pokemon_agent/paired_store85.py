"""Persist fully completed seat pairs so an interrupted long benchmark can resume."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import time

from .draft_league import read


def publish_json(destination, value):
    with NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".partial", dir=destination.parent, delete=False) as stream:
        partial = Path(stream.name)
        json.dump(value, stream, separators=(",", ":"))
        stream.write("\n")
    partial.rename(destination)


class PairedStore:
    def __init__(self, path, signature, pools, opponents):
        self.path = path
        self.pools = pools
        self.opponents = set(opponents)
        path.mkdir(parents=True, exist_ok=True)
        header = path / "manifest.json"
        expected = {"signature": signature, "pools": pools, "opponents": list(opponents)}
        if header.exists():
            saved = read(header)
            self.started = saved.pop("started")
            if saved != expected:
                raise ValueError("The saved pairs use a different candidate, pool draw, or opponent suite")
        else:
            self.started = time()
            publish_json(header, expected | {"started": self.started})

    def recover(self):
        records = {}
        cpu = 0.
        for file in self.path.glob("batch-*.json"):
            data = read(file)
            cpu += data["python_cpu"]
            for root, matches in data["records"].items():
                if root not in self.pools or int(root) in records or set(matches) != self.opponents:
                    raise ValueError("A saved batch has an unexpected pool or opponent")
                if any(len(pair) != 2 or {game["actor"] for game in pair} != {"p1", "p2"} for pair in matches.values()):
                    raise ValueError("A saved batch is missing one of its seats")
                records[int(root)] = matches
        return records, cpu

    def append(self, records, cpu):
        name = self.path / f"batch-{min(records)}.json"
        if name.exists():
            raise ValueError("A completed paired batch would be replaced")
        publish_json(name, {"records": records, "python_cpu": cpu})
