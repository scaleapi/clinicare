"""Repo-local overrides of harbor's OpenCode agent: skip the per-trial install, and keep
opencode's SQLite off the network FS.

TWO fixes, both for failure modes that cost whole sweeps.

1. INSTALL (see `install`). harbor reinstalls the harness inside every trial with
   `nvm install 22 && npm i -g opencode-ai@latest`. Node and opencode are now baked into the
   trial image (benchmark/_template/environment/Dockerfile), so that is 750 redundant network
   installs per sweep and the place sweeps die -- a gpt-oss run lost 1,169 attempts to exit 137
   during `npm i -g`. claude-code and codex avoid this because their harbor adapters check
   `command -v` and early-return; the opencode adapter has no such check, so we add one.

2. XDG (see `exec_as_agent`).

harbor's adapter hardcodes opencode's data dirs onto the collected-artifact mount:

    env["XDG_DATA_HOME"]  = "/logs/agent/opencode/xdg-data"
    env["XDG_STATE_HOME"] = "/logs/agent/opencode/xdg-state"

`/logs` is bind-mounted from the job dir, which on this host lives on EFS (NFS). opencode keeps a
per-session SQLite database there (`opencode.db` + `-wal` + `-shm`), and SQLite's WAL mode MMAPS
the `-shm` file. mmap over NFS cannot be backed reliably, and a write to an unbackable mapped page
raises SIGBUS -- which surfaces as Bun aborting the whole process:

    panic(main thread): Bus error at address 0x...
    oh no: Bun has crashed.

The agent then writes no report, opencode exits, and the trial scores 0 with no useful error. It
is load-dependent, so it hides: a 750-trial nemotron sweep saw 0 crashes on a quiet day, while a
muse sweep on a busy one lost ~48% of attempts (15 crashes / 31 scored), and the crashes hit EARLY
-- median 1 tool call before dying, against 30 for trials that survived.

This subclass redirects both dirs to container-local disk. The trade-off is deliberate: opencode's
internal DB and debug log are no longer collected into the job dir. Nothing scores off them --
`verifier/` and `agent/opencode.txt` (the parsed event stream) are unaffected, and those are what
grade.py and the trajectory parser read.

`exec_as_agent` is the hook rather than `run()` because `run()` is long and reimplementing it
would silently drift from upstream; this touches only the two variables and delegates the rest.
Delete once harbor makes the XDG location configurable.
"""

from __future__ import annotations

from typing import Any

from harbor.agents.installed.opencode import OpenCode

# Container-local, outside the /logs bind mount. /tmp is 1777 so the non-root `agent` user can
# create here.
#
# NOT `/tmp/opencode`: opencode's own `npm i -g` creates that directory ROOT-OWNED during the image
# build, so an agent-user mkdir under it fails with
#     EACCES: permission denied, mkdir '/tmp/opencode/xdg-data'
# and Bun exits before the agent runs. Keep this name distinct from anything the harness itself
# might squat on.
_LOCAL_XDG_DATA = "/tmp/oc-xdg/data"
_LOCAL_XDG_STATE = "/tmp/oc-xdg/state"


class OpenCodeLocalXDG(OpenCode):
    """OpenCode that uses the image's baked harness and keeps its SQLite off the network FS."""

    async def install(self, environment: Any) -> None:
        """Skip harbor's nvm+npm install when `opencode` is already on PATH.

        Mirrors what harbor's own claude-code and codex adapters do (`command -v` -> early
        return); the opencode adapter simply lacks the check. Falls through to the upstream
        install when the binary is absent, so this stays correct on an image without it baked.
        """
        probe = await self.exec_as_agent(
            environment, command="command -v opencode >/dev/null 2>&1 && echo PRESENT || true"
        )
        if "PRESENT" not in str(getattr(probe, "stdout", probe) or ""):
            await super().install(environment)
            return
        # Baked image: skip the nvm+npm download, but still run upstream's exec_as_root step.
        # Dropping it entirely broke the trial: harbor's first root exec is what prepares the
        # container for later privileged steps, and without it the verifier never produced
        # reward.json (every trial died RewardFileNotFoundError) even though test.sh runs fine
        # by hand in the same image. curl is already in the image, so this is a no-op apt call
        # kept for its side effects on the exec environment, not for the package.
        await self.exec_as_root(
            environment,
            command="apt-get update && apt-get install -y curl",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )

    async def exec_as_agent(  # type: ignore[override]
        self,
        environment: Any,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> Any:
        if env and ("XDG_DATA_HOME" in env or "XDG_STATE_HOME" in env):
            # Copy rather than mutate: the caller reuses this dict across several exec calls.
            env = {**env, "XDG_DATA_HOME": _LOCAL_XDG_DATA, "XDG_STATE_HOME": _LOCAL_XDG_STATE}
            # opencode does not create parents for these, and a missing dir is another early death.
            command = f"mkdir -p {_LOCAL_XDG_DATA} {_LOCAL_XDG_STATE}; {command}"
        return await super().exec_as_agent(
            environment, command, env=env, cwd=cwd, timeout_sec=timeout_sec
        )
