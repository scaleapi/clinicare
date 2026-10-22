"""Shared helpers for CliniCARE-Bench.

Exists so the things used from BOTH `benchmark/` and `construction/` have one home and one
copy, instead of being reached for with `sys.path.insert`. Most importantly `taskids`: a task id
is a hash of the cohort row, so two drifting copies would silently renumber every task.

Deliberately NOT in here:

  benchmark/_template/tests/*  — judge.py, evidence.py and policy_scoring.py run INSIDE the
      trial container, where `python3 judge.py` executes with no package installed and no
      third-party deps. That zero-dependency property is what makes the outcome score
      credential-free and reproducible, so the eval-side importers reach into `_template/tests`
      by path on purpose. Treating the container copy as the source of truth beats forking it.

  benchmark/scripts/env_image.py — content-hashes the Dockerfile and the mimic_tools sources by
      walking up from its own location. Repo-layout-aware code doesn't belong in a library, and
      it has a single importer sitting in the same directory.
"""

from .cohort import by_scenario, cohort_dir, load_dev, load_full, write_full
from .env import load_env
from .rewards import VERIFIER_FAILED_KEYS, verifier_failed, write_reward_atomic
from .taskids import param_hash, param_key, task_id

__all__ = [
    "VERIFIER_FAILED_KEYS",
    "by_scenario",
    "cohort_dir",
    "load_dev",
    "load_env",
    "load_full",
    "param_hash",
    "param_key",
    "task_id",
    "verifier_failed",
    "write_full",
    "write_reward_atomic",
]
