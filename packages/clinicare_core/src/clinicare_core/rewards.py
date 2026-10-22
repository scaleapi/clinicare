"""When a trial's reward.json is not a score.

The in-container verifier always writes a reward.json, so harbor has a reward to read. On the two
paths where it fails BEFORE scoring the outcome, that file still says `"score": 0.0`, but the
0.0 is not a verdict: nothing was checked. Every reader that turns rewards into numbers (the
eval reports, run_case.py and wrappers such as MedHELM) must treat those trials as unscored,
counted with the infrastructure failures, never as wrong answers. One copy of the rule, here, so
the readers cannot drift apart.

    rubrics_load_failed   judge.py could not load the rubric bundle (_template/tests/judge.py)
    judge_launch_failed   test.sh could not start judge.py at all (_template/tests/test.sh)

Rewriting a reward.json (the --amend-reward graders) goes through write_reward_atomic: a reader that
shares the job (another run_case.py or wrapper reusing the same finished trial) must never see the
file empty or half-written mid-update.

Deliberately NOT here:
    judge_failed          flags only the secondary process pass; the outcome is still scored.
    report_present == 0   the agent produced no deliverable: a genuine, scored 0.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path

VERIFIER_FAILED_KEYS = ("rubrics_load_failed", "judge_launch_failed")


def verifier_failed(reward: Mapping[str, object]) -> bool:
    """True when `reward` was written by a verifier that failed before scoring the outcome."""
    return any(reward.get(key) == 1 for key in VERIFIER_FAILED_KEYS)


def write_reward_atomic(path: Path | str, reward: Mapping[str, object]) -> None:
    """Replace `path` with `reward` as JSON so concurrent readers see the old file or the new one.

    `Path.write_text` truncates first, so a reader in between gets an empty or partial file. The
    new content goes to a temp file in the same directory (same filesystem, so the rename is
    atomic), then replaces the original in one step. The temp file takes the original's mode:
    mkstemp creates it 0600, which would leave a shared job's reward.json unreadable to other users.
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(reward, f, indent=2)
        if path.exists():
            os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
