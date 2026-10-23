"""CliniCARE evaluation sweep — run every (harness x model) system against one built task dir.

Fires each vendor-CLI system with its web tools disabled (this is a frozen-corpus eval; a live web
search would let an agent answer from outside the record). MIMIC is reached only through the
credential-isolated sidecar: the agent container holds no data credentials and proxies every
tool call over MIMIC_TOOLS_URL. Point --tasks at a policy / no-policy build to run that
ablation; the sweep itself is build-agnostic (build first with build_tasks.py).

Prereqs — from the repo root:
  bash scripts/build_image.sh                       # once, and after any image-input change
  set -a; source .env; set +a                       # plain KEY=VAL: must be EXPORTED, not sourced
  uv run python benchmark/scripts/build_tasks.py    # -> benchmark/tasks/

Needs MIMIC_API_SECRET, the vendor API key(s) for the systems you run, and JUDGE_* if you intend to
score the LLM-judged axes afterwards. See .env.template.

System groups: `--systems leaderboard` is the paper's 16 systems; `--systems ablation` is the ones used
only in the ablation sections. `--systems all` is both.

Backends (--backend, default docker):
  docker  : agent and sidecar both run locally; the sidecar is a compose service merged in with
            --extra-docker-compose. This is the supported path and needs no hosted infrastructure.
  modal   : agent in Modal cloud, reaching a sidecar you host yourself via MIMIC_TOOLS_URL. The
            compose overlay does not apply there, so the sidecar is your responsibility.

Usage:
  python scripts/run_sweep.py --dry-run                  # print harbor commands, run nothing
  python scripts/run_sweep.py --systems leaderboard         # the paper's 16 systems
  python scripts/run_sweep.py --systems claude-opus48,codex-gpt55
  python scripts/run_sweep.py --tasks benchmark/tasks_dev_policy  # the dev subset
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tomllib
import urllib.parse
from pathlib import Path
from typing import NamedTuple

BASE = Path(__file__).resolve().parent.parent
HARBOR = str(BASE / ".venv" / "bin" / "harbor")
PY = str(BASE / ".venv" / "bin" / "python")

# The trial image repo, for the stale-container check below. Imported from the one place that
# defines it so the two cannot drift.
sys.path.insert(0, str(BASE / "benchmark" / "scripts"))
from env_image import IMAGE_REPO as _IMAGE_REPO  # noqa: E402

WEB_OFF_CLAUDE = ["--ak", "disallowed_tools=WebSearch WebFetch"]
WEB_OFF_CODEX = ["--ak", "web_search=disabled"]  # off by default; set explicitly
# `--ak` cannot disable the Responses websocket: it only forwards keys declared as CLI_FLAGS
# descriptors and silently drops the rest. Use `CODEX_HTTP_ONLY=1` (see codex_mcp_fix).

# gemini web is stripped by the GeminiNoWeb agent itself -- no flag needed.

# opencode is the only harness with no MCP timeout raised elsewhere, and the only one that loses
# tool calls to `MCP error -32001: Request timed out`. A timed-out call returns no data and the
# agent then answers without evidence, so this is a correctness fix, not a speed one.
# `experimental.mcp_timeout` is the knob that governs tool EXECUTION; the schema-documented
# per-server `mcp.<name>.timeout` only affects tool DISCOVERY and is ignored for execution.
# `provider.openai.options.timeout: false` stops the model layer re-imposing its own ceiling.
# 600000 ms matches claude-code's MCP_TOOL_TIMEOUT so the harnesses are on equal footing.
WEB_OFF_OPENCODE = [
    "--ak",
    'opencode_config={"tools":{"webfetch":false},'
    '"experimental":{"mcp_timeout":600000},'
    '"provider":{"openai":{"options":{"timeout":false}}}}',
]


# Which env vars each harness family reads. One source: the credential forwarding in creds() and
# the allowlist preflight below must agree on the base-URL var, or the preflight checks a variable
# nobody sends.
API_KEY_VAR = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
BASE_URL_VAR = {
    "anthropic": "ANTHROPIC_BASE_URL",
    "openai": "OPENAI_BASE_URL",
    "gemini": "GOOGLE_GEMINI_BASE_URL",
}


def gateway_not_allowlisted(tasks: str, systems: list[System]) -> list[tuple[str, str]]:
    """[(env var, host)] for every *_BASE_URL a system would use whose host the built tasks don't allow.

    allowed_hosts is baked in at BUILD time (Harbor won't interpolate it), so exporting a gateway
    URL after building silently produces tasks that can't reach it. The agent then surfaces the
    block as a TLS/certificate failure minutes later rather than a refusal, which sends you looking
    at CA bundles instead of the allowlist. Cheap to check, so check it.
    """
    task_tomls = sorted((BASE / tasks).glob("*/task.toml"))
    if not task_tomls:
        return []  # nothing built yet; the task-dir check above already covers that
    allowed = set(
        tomllib.loads(task_tomls[0].read_text()).get("agent", {}).get("allowed_hosts", [])
    )
    if not allowed:
        return []  # no allowlist on this build (network_mode != allowlist) -> nothing to enforce
    bad = []
    for base_var in sorted({BASE_URL_VAR[a.creds] for a in systems}):
        url = os.environ.get(base_var, "").strip()
        if not url:
            continue  # unset -> vendor default, which is allowlisted by build_tasks
        host = urllib.parse.urlsplit(url).hostname or ""
        if host and host not in allowed:
            bad.append((base_var, host))
    return bad


_WARNED_KEYS: set[str] = set()


def creds(family: str, dry_run: bool = False) -> list[str]:
    """`--ae KEY=VAL` pairs for a harness family.

    Each vendor CLI reads its own key. A base URL is forwarded ONLY when the operator set one, so
    the default is the vendor's own endpoint rather than any gateway. build_tasks.py allowlists a
    set gateway's host automatically, so pointing a system at one needs no extra config — but the
    allowlist is baked in at BUILD time, so rebuild the tasks after changing a base URL (the
    preflight in main() catches that drift). The one thing still on you: the route suffix differs
    by client — Claude Code appends /v1/messages and wants no /v1 on the base; OpenAI-compatible
    clients want it.
    """
    key_var, base_var = API_KEY_VAR[family], BASE_URL_VAR[family]
    key = os.environ.get(key_var, "")
    if not key:
        # --dry-run is the "is my setup right?" command, so it has to survive missing keys and
        # report ALL of them rather than exiting on the first. The placeholder never leaves the
        # printed command (the mask below catches it too).
        if dry_run:
            if key_var not in _WARNED_KEYS:  # once per key, not once per system
                _WARNED_KEYS.add(key_var)
                print(
                    f"  ! {key_var} is not set — required for the {family} systems", file=sys.stderr
                )
            key = f"<unset:{key_var}>"
        else:
            sys.exit(
                f"{key_var} is not set — required for the {family} systems. See .env.template."
            )
    pairs = [f"{key_var}={key}"]
    if os.environ.get(base_var):
        pairs.append(f"{base_var}={os.environ[base_var]}")
    return [x for kv in pairs for x in ("--ae", kv)]


class System(NamedTuple):
    tag: str  # short id -> job name cbench_sweep_<tag>
    group: str  # "leaderboard" (the paper's 16 systems) or "ablation"
    agent: list[str]  # harbor -a args: a built-in name, or a module:Class import path
    model: str  # provider-qualified model id passed to harbor -m
    creds: str  # cred family: "anthropic" | "openai" | "gemini"
    web_off: list[str]  # web-disable --ak args ([] when the agent handles it, e.g. GeminiNoWeb)


# NB: confirm the frontier model ids are live on the proxy before the real run (see --dry-run).
SYSTEMS: list[System] = [
    # Each vendor model on its NATIVE CLI, web disabled per harness. Two groups:
    #   leaderboard — the 16 systems in the paper's main table: 10 production CLIs on their own
    #                 vendor models, plus 6 open-weight models on opencode, the neutral harness.
    #   ablation    — systems that appear only in the ablation sections (policy on/off, repeated-run
    #                 reliability, reasoning effort, and the fixed-model harness comparison).
    #
    # THE OPEN-WEIGHT SYSTEMS NEED A GATEWAY. Their model ids (`openai/fireworks_ai/...`) are routed
    # by an OpenAI-compatible proxy, not by any vendor endpoint, so set OPENAI_BASE_URL to your own
    # and build the tasks with it set — build_tasks.py allowlists that host for you.
    # Confirm a model id is live with your provider before a real run (see --dry-run).
    System(
        "claude-opus5",
        "leaderboard",
        ["-a", "claude-code"],
        "claude-opus-5",
        "anthropic",
        WEB_OFF_CLAUDE,
    ),
    System(
        "claude-sonnet5",
        "leaderboard",
        ["-a", "claude-code"],
        "claude-sonnet-5",
        "anthropic",
        WEB_OFF_CLAUDE,
    ),
    System(
        "codex-gpt55",
        "leaderboard",
        ["-a", "clinicare_core.agents.codex_mcp_fix:CodexMcpFixed"],
        "openai/gpt-5.5",
        "openai",
        WEB_OFF_CODEX,
    ),
    System(
        "codex-gpt56sol",
        "leaderboard",
        ["-a", "clinicare_core.agents.codex_mcp_fix:CodexMcpFixed"],
        "openai/gpt-5.6-sol",
        "openai",
        WEB_OFF_CODEX,
    ),
    System(
        "codex-gpt54",
        "leaderboard",
        ["-a", "clinicare_core.agents.codex_mcp_fix:CodexMcpFixed"],
        "openai/gpt-5.4",
        "openai",
        WEB_OFF_CODEX,
    ),
    System(
        "gemini-31pro",
        "leaderboard",
        ["-a", "clinicare_core.agents.gemini_no_web:GeminiNoWeb"],
        "gemini/gemini-3.1-pro-preview",
        "gemini",
        [],
    ),
    System(
        "codex-gpt54mini",
        "leaderboard",
        ["-a", "clinicare_core.agents.codex_mcp_fix:CodexMcpFixed"],
        "openai/gpt-5.4-mini",
        "openai",
        WEB_OFF_CODEX,
    ),
    System(
        "codex-gpt56luna",
        "leaderboard",
        ["-a", "clinicare_core.agents.codex_mcp_fix:CodexMcpFixed"],
        "openai/gpt-5.6-luna",
        "openai",
        WEB_OFF_CODEX,
    ),
    System(
        "gemini-35flash",
        "leaderboard",
        ["-a", "clinicare_core.agents.gemini_no_web:GeminiNoWeb"],
        "gemini/gemini-3.5-flash",
        "gemini",
        [],
    ),
    System(
        "gemini-36flash",
        "leaderboard",
        ["-a", "clinicare_core.agents.gemini_no_web:GeminiNoWeb"],
        "gemini/gemini-3.6-flash",
        "gemini",
        [],
    ),
    # --- opencode: the six open-weight models, the paper's controlled comparison ---
    # All six run the repo-local OpenCodeLocalXDG shim, not harbor's stock `opencode`, for the same
    # kind of reason the codex arms run CodexMcpFixed. The stock adapter (a) reinstalls the harness
    # inside every trial via nvm + `npm i -g opencode-ai`, which is redundant now that the binary is
    # baked into the trial image and is where a gpt-oss sweep lost 1,169 attempts to exit 137, and
    # (b) hardcodes opencode's XDG dirs onto the /logs bind mount. /logs is the job dir, which is on
    # EFS here; opencode keeps a SQLite session DB there and WAL mode mmaps its -shm file, which over
    # NFS raises SIGBUS -> "Bun has crashed" -> empty report -> a silent score of 0 (~48% of one muse
    # sweep). Reverting any of these to a bare `opencode` silently reinstates both failure modes,
    # and neither announces itself in the metrics -- so keep them in step with the shim.
    System(
        "opencode-ds4pro",
        "leaderboard",
        ["-a", "clinicare_core.agents.opencode_local_xdg:OpenCodeLocalXDG"],
        "openai/fireworks_ai/deepseek-v4-pro",
        "openai",
        WEB_OFF_OPENCODE,
    ),
    System(
        "opencode-ds4flash",
        "leaderboard",
        ["-a", "clinicare_core.agents.opencode_local_xdg:OpenCodeLocalXDG"],
        "openai/fireworks_ai/deepseek-v4-flash",
        "openai",
        WEB_OFF_OPENCODE,
    ),
    System(
        "opencode-qwen37",
        "leaderboard",
        ["-a", "clinicare_core.agents.opencode_local_xdg:OpenCodeLocalXDG"],
        "openai/fireworks_ai/qwen3p7-plus",
        "openai",
        WEB_OFF_OPENCODE,
    ),
    System(
        "opencode-glm52",
        "leaderboard",
        ["-a", "clinicare_core.agents.opencode_local_xdg:OpenCodeLocalXDG"],
        "openai/fireworks_ai/glm-5p2",
        "openai",
        WEB_OFF_OPENCODE,
    ),
    System(
        "opencode-minimaxm3",
        "leaderboard",
        ["-a", "clinicare_core.agents.opencode_local_xdg:OpenCodeLocalXDG"],
        "openai/fireworks_ai/minimax-m3",
        "openai",
        WEB_OFF_OPENCODE,
    ),
    System(
        "opencode-kimik2p7",
        "leaderboard",
        ["-a", "clinicare_core.agents.opencode_local_xdg:OpenCodeLocalXDG"],
        "openai/fireworks_ai/kimi-k2p7-code",
        "openai",
        WEB_OFF_OPENCODE,
    ),
    # --- ablation-only systems (not on the leaderboard) ---
    # Opus 4.8 carries the policy-corpus, repeated-run reliability and reasoning-effort ablations.
    System(
        "claude-opus48",
        "ablation",
        ["-a", "claude-code"],
        "claude-opus-4-8",
        "anthropic",
        WEB_OFF_CLAUDE,
    ),
    # The harness comparison holds DeepSeek-V4-Pro fixed across three model-agnostic harnesses;
    # opencode-ds4pro above is the third.
    System(
        "qwen-ds4pro",
        "ablation",
        ["-a", "qwen-coder"],
        "fireworks_ai/deepseek-v4-pro",
        "openai",
        [],
    ),
    System(
        "miniswe-ds4pro",
        "ablation",
        ["-a", "mini-swe-agent"],
        "openai/fireworks_ai/deepseek-v4-pro",
        "openai",
        [],
    ),
]


def job_name(tag: str, tasks: str) -> str:
    """`cbench_sweep_<tag>` for a baseline build; a variant task dir contributes its suffix, so
    variant runs don't collide with the baseline job and their scores pool separately (task ids
    are shared, so they still pool per-task within a variant).

    build_tasks.py produces `tasks/` and `tasks_answer/`; build_dev148.py produces
    `tasks_dev_policy/` and `tasks_dev_nopolicy/`. ORDER MATTERS below — longest prefix first,
    and only prefixes that should yield an EMPTY suffix belong in the list. `tasks_dev` must
    precede `tasks`, or `tasks_dev_nopolicy` matches `tasks` and yields `_dev_nopolicy` instead
    of `_nopolicy`. Do not add `tasks_answer`: it must fall through to `tasks` so it keeps the
    `_answer` suffix rather than colliding with the baseline job name.
    """
    leaf = os.path.basename(tasks.rstrip("/"))
    for base in ("tasks_dev_policy", "tasks_dev", "tasks"):
        if leaf.startswith(base):
            suffix = leaf[len(base) :]  # "" for the base build; "_scaffold"/"_md"/… for ablations
            break
    else:
        suffix = "_" + leaf
    return f"cbench_sweep_{tag}{suffix}"


def build_cmd(
    system: System,
    tasks: str,
    k: int,
    n: int,
    backend: str,
    effort: str | None = None,
    gen: str = "",
    max_retries: int | None = None,
    dry_run: bool = False,
    job: str | None = None,
    extra: tuple[str, ...] = (),
) -> list[str]:
    """`job` overrides the derived job name; `extra` is appended verbatim (run_case.py passes
    `-i <task id> -o <jobs dir>` to run one task of a dataset into a caller-chosen job dir)."""
    cmd = [
        HARBOR,
        "run",
        "-p",
        tasks,
        *system.agent,
        "-m",
        system.model,
        "-e",
        backend,
        # Agent setup (nvm/Node + `npm i -g` for gemini-cli) can exceed the
        # 360s default when many trials install at once -> setup-timeout failures. 3x (1080s) absorbs
        # the contention; harmless for fast-setup systems (it's a ceiling, not a fixed wait).
        "--agent-setup-timeout-multiplier",
        "3",
    ]
    if backend != "modal":
        # docker: run the credentialed sidecar locally as a compose service. It bind-mounts the
        # MIMIC Parquet tree read-only and is the only container that touches patient data. The
        # agent reaches it at http://mimic-tools.internal:8799 — the compose network alias, which
        # is dotted because some backends reject a bare label in allowed_hosts (see task.toml).
        # On modal you host the sidecar yourself and point MIMIC_TOOLS_URL at it.
        cmd += ["--extra-docker-compose", f"{tasks}/_sidecar.compose.yml"]
    # Sidecar creds (MIMIC_TOOLS_URL / MIMIC_API_SECRET) reach the container via the task's
    # [environment.env] ${VAR} substitution from the sourced env -- not via --ae.
    # reasoning-effort axis: harbor passes it as an agent kwarg (codex -> -c model_reasoning_effort,
    # gemini -> thinkingConfig, claude-code -> --effort). Suffix the job name so effort runs don't
    # collide with the baseline job (and pool separately when scored).
    jn = job or job_name(system.tag, tasks) + gen + (f"_eff-{effort}" if effort else "")
    cmd += [
        *creds(system.creds, dry_run),
        *system.web_off,
        "--job-name",
        jn,
        "-k",
        str(k),
        "-n",
        str(n),
        "--no-delete",
        "--yes",
    ]
    if effort:
        cmd += ["--ak", f"reasoning_effort={effort}"]
    # Retry budget. Harbor's default here is 0, so EVERY death is permanent and becomes manual gap-build
    # work. That default is right for a headline sweep (a retry changes the per-task attempt count) but
    # wrong for a RECOVERY build, where the deaths being repaired are transient by construction.
    # `ApiRateLimitError` is NOT in this job's `retry.exclude_exceptions`
    # (RewardFileEmpty/VerifierTimeout/AgentTimeout/RewardFileNotFound/ApiUsageLimit/VerifierOutputParse),
    # so a non-zero budget recovers exactly the 429 burst class -- while `AgentTimeoutError` stays
    # excluded, so a genuine 5400 s ceiling hit is still terminal and is NOT silently re-attempted.
    if max_retries is not None:
        cmd += ["--max-retries", str(max_retries)]
    return cmd + list(extra)


def select(spec: str) -> list[System]:
    if spec == "all":
        return SYSTEMS
    tags = set(spec.split(","))
    return [a for a in SYSTEMS if a.group == spec or a.tag in tags]


def preflight(systems: list[System], tasks: str, backend: str) -> int:
    """Refuse a run that cannot work (missing key/secret, unbuilt tasks, a gateway the build
    can't reach) and warn about orphaned sidecars. 0 = go, 1 = refused (message on stderr).
    Shared by run_sweep.py and run_case.py; callers skip it on --dry-run.
    """
    # Check only the keys the SELECTED systems actually need — requiring all three would stop
    # someone running a single-vendor sweep.
    needed = {"MIMIC_API_SECRET"} | {
        {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }[a.creds]
        for a in systems
    }
    for var in sorted(needed):
        if not os.environ.get(var):
            print(
                f"{var} not set -- `set -a; source .env; set +a` at the repo root first "
                "(the file is plain KEY=VAL, so it must be exported)",
                file=sys.stderr,
            )
            return 1
    if not (BASE / tasks).is_dir():
        print(
            f"task dir not found: {tasks} -- build it with build_tasks.py first",
            file=sys.stderr,
        )
        return 1
    if unreachable := gateway_not_allowlisted(tasks, systems):
        for var, host in unreachable:
            print(
                f"{var} points at {host}, which is NOT in the allowed_hosts baked into "
                f"{tasks} — that build predates the current {var}. The egress sidecar would "
                f"block it, and the agent reports a misleading TLS error after a multi-minute "
                f"hang rather than a refusal.\n"
                f"  Rebuild with {var} set as it is now (build_tasks allowlists it automatically):\n"
                f"    uv run python benchmark/scripts/build_tasks.py",
                file=sys.stderr,
            )
        return 1
    # Pre-flight orphan guard (docker only): stale sidecar containers from failed teardowns pile up and
    # starve new trials (RuntimeError / "cannot stop container"). Warn if many are already live so we
    # clean first. Root trigger is too much concurrency -> keep to <=2 sweeps at once and a modest -n.
    if backend == "docker":
        try:
            imgs = subprocess.run(
                ["docker", "ps", "--format", "{{.Image}}"],
                capture_output=True,
                text=True,
                timeout=15,
            ).stdout
            stale = sum(
                1
                for l in imgs.splitlines()
                # Match the configured image repo, not a hardcoded name: env_image resolves it
                # (CLINICARE_IMAGE_REPO), and a stale literal here makes this check silently
                # never fire rather than error.
                if _IMAGE_REPO in l or "egress-control-sidecar" in l
            )
            if stale > 24:
                print(
                    f"WARNING: {stale} mimic/egress sidecar containers already running -- likely "
                    f"orphans that will starve this sweep. Clear them with\n"
                    f"    docker rm -f $(docker ps -q --filter ancestor={_IMAGE_REPO})\n"
                    f"  and avoid >2 concurrent sweeps / high -n.",
                    file=sys.stderr,
                )
        except Exception:
            pass

    return 0


def child_env(backend: str) -> dict[str, str] | None:
    """The environment harbor runs with, or None (message on stderr) if the backend can't run."""
    env = os.environ.copy()
    if backend == "modal":
        # There is no default sidecar to fall back on. The compose overlay does not apply on
        # modal, so you must host one yourself and point MIMIC_TOOLS_URL at it. Refusing here
        # beats launching every trial only for each to die at its first tool call.
        if not env.get("MIMIC_TOOLS_URL"):
            print(
                "MIMIC_TOOLS_URL is not set. The modal backend cannot start the sidecar for you "
                "(the docker-compose overlay applies only to --backend docker) — host one and "
                "point MIMIC_TOOLS_URL at it, or use --backend docker. See "
                "packages/mimic_tools/SIDECAR.md.",
                file=sys.stderr,
            )
            return None
    else:
        env.pop(
            "MIMIC_TOOLS_URL", None
        )  # docker: the compose overlay hardcodes the local sidecar (mimic-tools.internal:8799), so
        # a stray .env value can't redirect a "local" run at the DEPLOYED sidecar. Popping keeps the
        # docker-compose interpolation environment clean too.

    return env


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tasks", default="benchmark/tasks", help="built task dir to run against")
    ap.add_argument(
        "--systems", default="all", help="all | leaderboard | ablation | comma-sep tags"
    )
    ap.add_argument(
        "--backend", choices=["docker", "modal"], default="docker", help="harbor -e backend"
    )
    ap.add_argument("-k", type=int, default=1, help="attempts per task")
    ap.add_argument("-n", type=int, default=8, help="concurrent trials")
    ap.add_argument(
        "--job-suffix",
        default="",
        help="tag appended to every job name for this sweep (e.g. --job-suffix v2). Use it to keep a "
        "new experiment generation separate from earlier results after a prompt/judge change, "
        "so nothing pools across incomparable conditions.",
    )
    ap.add_argument(
        "--effort",
        choices=["minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="reasoning-effort level (vendor CLI systems only); job names get an _eff-<level> suffix so they pool separately",
    )
    ap.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="harbor -r: retry budget per trial. Omitted => harbor's default of 0 (every death "
        "permanent). Use a non-zero value for RECOVERY builds: ApiRateLimitError is not in "
        "retry.exclude_exceptions so it is recovered, while AgentTimeoutError still is, so a real "
        "agent-ceiling hit stays terminal.",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the harbor commands, run nothing")
    ap.add_argument(
        "--parallel",
        action="store_true",
        help="run all systems concurrently (one harbor process each) instead of sequentially",
    )
    ap.add_argument(
        "--score", action="store_true", help="score all selected systems into one CSV after running"
    )
    ap.add_argument("--csv", default="sweep_scores.csv", help="combined scores CSV")
    args = ap.parse_args()
    os.chdir(BASE)

    systems = select(args.systems)
    if not systems:
        print(f"no systems match {args.systems!r}", file=sys.stderr)
        return 1
    if not args.dry_run and (rc := preflight(systems, args.tasks, args.backend)):
        return rc
    env = child_env(args.backend)
    if env is None:
        return 1

    # generation tag first, then effort -> cbench_sweep_<system>[_<variant>]_<gen>[_eff-<level>]
    gen_sfx = ("_" + args.job_suffix.lstrip("_")) if args.job_suffix else ""
    eff_sfx = gen_sfx + (f"_eff-{args.effort}" if args.effort else "")
    procs = []  # (system, Popen, filehandle) -- populated only under --parallel
    failed = False  # any system's non-zero exit -> reflected in the process return code
    for system in systems:
        cmd = build_cmd(
            system,
            args.tasks,
            args.k,
            args.n,
            args.backend,
            args.effort,
            gen_sfx,
            args.max_retries,
            args.dry_run,
        )
        print(f"### {system.tag:20s} -> {system.model}")
        if args.dry_run:
            shown = [
                c.split("=")[0] + "=***" if "_API_KEY=" in c else c for c in cmd
            ]  # mask secrets
            print("    " + " ".join(shlex.quote(c) for c in shown))
            continue
        log = BASE / "jobs" / f"{job_name(system.tag, args.tasks)}{eff_sfx}.log"
        log.parent.mkdir(exist_ok=True)
        fh = open(log, "w")
        if args.parallel:  # launch all systems at once, wait below
            procs.append(
                (system, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env), fh)
            )
        else:  # sequential: one system at a time
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
            fh.close()
            failed = failed or rc != 0
            print(f"    exit={rc} - log={log.relative_to(BASE)}")
    for system, p, fh in procs:
        rc = p.wait()
        fh.close()
        failed = failed or rc != 0
        print(
            f"### {system.tag:20s} exit={rc} - log=jobs/{job_name(system.tag, args.tasks)}{eff_sfx}.log"
        )

    if args.score and not args.dry_run:
        job_dirs = [f"jobs/{job_name(a.tag, args.tasks)}{eff_sfx}" for a in systems]
        subprocess.run(
            [PY, "benchmark/eval/report/score_clinicare.py", *job_dirs, "--csv", args.csv],
            env=env,
        )
        print(f"scores -> {args.csv}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
