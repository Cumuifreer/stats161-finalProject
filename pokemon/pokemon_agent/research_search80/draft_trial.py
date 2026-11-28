"""Run the public Draft completion portfolio through paired development matches."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
from unittest.mock import patch

from .. import blind85, lab85, quicksilver80
from .draft_mix import MixedDraft, PublicDraftWorlds


def evaluate(roots, cards, rivals, checkout, variant, preview, lineup_model, value, draft_review,
             full_preview, lead, action, deep, quicksilver, settings):
    if not settings.get("draft_bundles"):
        raise ValueError("The Draft portfolio augments the frozen bundle preselector")
    with PublicDraftWorlds(quicksilver) as simulator, patch.object(lab85, "DraftBundles", partial(MixedDraft, simulator=simulator)):
        return lab85.evaluate(roots, cards, rivals, checkout, variant, preview, lineup_model, value, draft_review,
                              full_preview, lead, action, deep, quicksilver, settings)


def measure(roots, cards, rivals, checkout, variant, workers, *args):
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        jobs = [pool.submit(evaluate, [root], cards, rivals, checkout, variant, *args) for root in roots]
        for job in as_completed(jobs):
            yield job.result()


def freeze(*args):
    result = FROZEN(*args)
    for source in (Path(__file__), Path(__file__).with_name("draft_mix.py")):
        result["files"][str(source)] = sha256(source.read_bytes()).hexdigest()
    return result


FROZEN = quicksilver80.freeze


if __name__ == "__main__":
    with patch.object(blind85, "measure", measure), patch.object(quicksilver80, "freeze", freeze):
        quicksilver80.main()
