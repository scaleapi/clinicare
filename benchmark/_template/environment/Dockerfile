# Trial container image for CliniCARE-Bench — the build recipe for the prebuilt image the tasks
# reference via [environment].docker_image.
#
# Kept under _template/environment/ per Harbor's convention (a task's environment is defined
# here). There is no external base to inherit from, so this assembles one from two independent
# concerns:
#   1. the canonical MIMIC tool surface + stdio MCP server + FastAPI service + DuckDB backend,
#      from the `mimic_tools` library (installed as `mimic-tools[mcp,api]`);
#   2. `openai`, which the in-trial verifier does NOT need (it is deterministic — no LLM, no
#      credentials, no network). It is kept so the post-hoc scorers can run in this image if
#      desired; ../tests/judge.py and ../tests/policy_scoring.py import the OpenAI SDK lazily,
#      on the offline path only.
#
# The judge dep lives HERE, not in `mimic_tools`: the library never imports openai (it is a
# pure tool surface consumed by the console / API / MCP server), so its package metadata
# stays clean. openai is a verifier concern, so it is layered into the benchmark's image.
#
# Because assembling the base needs the mimic_tools source (the COPYs below), this image is
# built OUT OF BAND — `bash scripts/build_image.sh` builds it with the mimic_tools package as
# the build context (`docker build -f <this file> packages/mimic_tools`) into your local daemon.
# The tasks reference it by ref (docker_image), so Harbor reuses it rather than rebuilding per
# trial (a per-task `FROM`-build can't see the mimic_tools source). A local image is enough;
# push to a registry only if a remote backend has to pull it (build_image.sh --push).
#
# It ships the claude-code AND codex CLIs baked in (see the bottom of this file). Harbor normally
# installs the agent at runtime, but we bake both because their runtime installs are either broken
# or heavy/flaky here, and because both adapters early-return when the binary is already on PATH:
#   - claude-code: its runtime installer runs `claude install latest`, whose native-build self-
#     install step is SIGKILL'd (exit 137) in this container (not OOM — 16 GB still 137s) — while
#     the binary itself runs fine. Harbor detects `claude` on PATH (`command -v claude`) and SKIPS
#     the broken install. So NO `--agent-kwarg version` is needed; `-a claude-code` just works.
#   - codex: its glibc runtime path is heavy + internet-dependent (nvm bootstrap → Node 22 →
#     `npm install -g @openai/codex`). Upstream ships a static-musl binary per arch, so we bake it
#     the same way. Harbor's codex adapter checks `command -v codex` and SKIPS the install — the
#     natural `-m openai/gpt-5.5` invocation (via our CodexMcpFixed adapter) just works, no version
#     kwarg. NOTE: that early-return also skips codex's own `apt-get install curl ripgrep`, so we
#     bake `ripgrep` into the apt layer below (codex shells out to `rg`; curl is already present).
# gemini-cli is still installed at runtime: it is npm-only (needs a Node runtime) AND its Harbor
# adapter has no "already installed" check, so a baked binary would be reinstalled every trial
# anyway — baking it buys nothing without an upstream/adapter change.
#
# The agent reaches the tools over an MCP stdio server (`python -m mimic_tools.mcp_server`) in
# PROXY mode, declared in each task's task.toml [[environment.mcp_servers]]: it holds no creds and
# forwards every call to the mimic-tools sidecar. The SIDECAR (`python -m mimic_tools.serve`, the
# SAME image, different command) is the only process that reads MIMIC — from a read-only bind
# mount of the local Parquet tree. It needs no DuckDB extensions and no network: local Parquet is
# a plain file read, so nothing beyond the `duckdb` wheel installed below is required.
#
# Build with: bash scripts/build_image.sh
FROM python:3.11-slim

# curl + ca-certificates: fetch the baked agent binaries below (and general TLS). ripgrep: codex
# shells out to `rg`; its runtime install would apt-get it, but the baked-binary early-return skips
# that step, so we supply it here.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates ripgrep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/mimic-tools
COPY pyproject.toml README.md ./
COPY src ./src
# [mcp] = the stdio server the agent runs (proxy mode — forwards to the sidecar via MIMIC_TOOLS_URL);
# [api] = the FastAPI service the credentialed sidecar runs (python -m mimic_tools.serve).
# One image serves both roles — the agent container and the mimic-tools sidecar differ only by
# command + env, not by image. See packages/mimic_tools/SIDECAR.md.
RUN pip install --no-cache-dir ".[mcp,api]"

# Verifier dependency (this domain's process-rubric judge), NOT a mimic_tools dependency.
RUN pip install --no-cache-dir "openai>=1.0.0"

# Agent analytics toolkit for the Shell Execution surface — the libraries the agent may import
# in its Python scratchpad for timeline reconstruction, clinical-score / interval computation,
# and structured intermediate artifacts. Baked in (NOT pip-installed at runtime) because the
# governed setup denies the agent PyPI egress (network_mode = "allowlist"; see mimic_tools/
# SIDECAR.md), so anything not in the image is unavailable. python-dateutil arrives via pandas;
# named here so the version is explicit. All ship manylinux wheels for amd64 + arm64 (no build
# toolchain needed on the slim base). For a frozen benchmark release, replace the floors with
# exact `==` pins captured from a known-good build (`pip freeze`) so the image is reproducible.
RUN pip install --no-cache-dir \
        "numpy>=2.0" \
        "pandas>=2.2" \
        "scipy>=1.13" \
        "python-dateutil>=2.9" \
        "tabulate>=0.9"

# Claude Code, baked in. We place the native binary on PATH directly rather than running the
# usual `bootstrap.sh | bash`, because that installer's `claude install latest` step is killed
# (exit 137) inside this container even though the binary runs fine. With the binary on PATH,
# Harbor detects it (`command -v claude`) and skips its own install — no `--agent-kwarg version`
# needed. CLAUDE_CODE_VERSION here is the single source of truth for the version; bump to upgrade.
# TARGETARCH is provided by buildx per platform; no `claude --version` here (the arm64 leg would
# run under QEMU) — it's verified at trial time on the native arch.
ARG CLAUDE_CODE_VERSION=2.1.205
ARG TARGETARCH
RUN case "${TARGETARCH:-amd64}" in \
        amd64) CC_ARCH=x64 ;; \
        arm64) CC_ARCH=arm64 ;; \
        *) echo "unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && curl -fsSL "https://downloads.claude.ai/claude-code-releases/${CLAUDE_CODE_VERSION}/linux-${CC_ARCH}/claude" \
         -o /usr/local/bin/claude \
    && chmod +x /usr/local/bin/claude

# Codex CLI, baked in — same pattern as claude-code. Upstream ships a static-musl single binary per
# arch (runs on this glibc base; static-pie, no libc dep) as a GitHub release asset named
# `codex-<rust-target>.tar.gz`, whose sole entry is the arch-named binary we rename to `codex`.
# Release tag is `rust-v${CODEX_VERSION}`. With the binary on PATH, Harbor's codex adapter's
# `command -v codex` check skips its (heavy nvm+npm) runtime install — no `--agent-kwarg version`.
# CODEX_VERSION is the single source of truth; bump to upgrade. TARGETARCH is provided by buildx.
ARG CODEX_VERSION=0.144.6
RUN case "${TARGETARCH:-amd64}" in \
        amd64) CX_TARGET=x86_64-unknown-linux-musl ;; \
        arm64) CX_TARGET=aarch64-unknown-linux-musl ;; \
        *) echo "unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && curl -fsSL "https://github.com/openai/codex/releases/download/rust-v${CODEX_VERSION}/codex-${CX_TARGET}.tar.gz" \
         -o /tmp/codex.tar.gz \
    && tar -xzf /tmp/codex.tar.gz -C /usr/local/bin \
    && mv "/usr/local/bin/codex-${CX_TARGET}" /usr/local/bin/codex \
    && chmod +x /usr/local/bin/codex \
    && rm -f /tmp/codex.tar.gz

# Non-root agent user. The agent container is the sandbox for an UNTRUSTED LLM with shell
# access, so it must not run as uid 0: any container-escape primitive (a permissive bind mount,
# a kernel bug, a reachable docker socket) becomes host root instead of host nobody. Third
# parties run this benchmark on their own infrastructure against credentialed data.
#
# Three things make this safe rather than a foot-gun:
#   - Everything above installs as root into /usr/local and /opt, so site-packages and the baked
#     CLIs stay root-owned and read-only to the agent — it cannot patch its own tool surface.
#   - The CLIs only need a writable $HOME (~/.claude, ~/.codex, ~/.gemini, and the ~/.nvm that
#     gemini-cli's runtime install bootstraps), which `useradd -m` provides. Harbor's own
#     install steps use exec_as_root, so `apt-get`/npm-prefix work regardless of this USER.
#   - task.toml sets [agent].user to match. That is not redundant: Harbor only chowns
#     files it uploads (auth creds, task inputs) when `environment.default_user` is set, and
#     that comes from the task config, NOT from this directive. Set one without the other and
#     uploads land root-owned in a container the agent cannot write — keep them in lockstep.
#
# The mimic-tools SIDECAR runs this same image and deliberately overrides back to root
# (sidecar.compose.yml), because it must read the operator's host Parquet tree through a bind
# mount whose uid we do not control. It holds no shell for the agent — only the HTTP tool
# surface — so it is not the container this hardening is about.
# Node + opencode, baked in — same reasoning as claude-code and codex above, and for a sharper
# reason. harbor's opencode adapter installs the harness INSIDE EVERY TRIAL:
#
#     nvm install 22 && npm i -g opencode-ai@latest
#
# On a 750-task sweep that is 750 Node downloads and 750 npm resolutions. It is the only remaining
# per-trial network install in the image, and it is where sweeps die: a gpt-oss run lost 1,169
# attempts to `exit 137` (SIGKILL) during `npm i -g`, with the host showing 1.7 TB free memory,
# 4% inode use and low memory pressure — i.e. not a host OOM, but a failure mode we could not
# pin down and could not stop hitting. Baking removes the variable rather than scheduling around it.
#
# It also fixes reproducibility: `@latest` is unpinned, so the harness silently drifted mid-project
# (nemotron's 750 ran opencode 1.18.21; muse's ran 1.18.23). OPENCODE_VERSION below is now the
# single source of truth — bump it deliberately, the way CLAUDE_CODE_VERSION and CODEX_VERSION are.
#
# The .tar.gz build is deliberate: python:3.11-slim ships no xz-utils, so `tar -J` fails.
# Installed as ROOT into /usr/local (before `USER agent`), so the agent cannot patch its own
# harness — the same property the comment below claims for the other CLIs. harbor's adapter has no
# "already installed" check of its own, unlike claude-code/codex, so the repo-local
# OpenCodeLocalXDG shim overrides install() to skip when `opencode` is already on PATH.
ARG NODE_VERSION=22.23.2
ARG OPENCODE_VERSION=1.18.23
RUN case "${TARGETARCH:-amd64}" in \
        amd64) NODE_ARCH=x64 ;; \
        arm64) NODE_ARCH=arm64 ;; \
        *) echo "unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${NODE_ARCH}.tar.gz" \
         -o /tmp/node.tar.gz \
    # --no-same-owner is REQUIRED: the Node tarball carries a numeric uid/gid that does not
    # exist here, and tar-as-root preserves it, leaving /usr/local/{bin,lib,share} owned by
    # UNKNOWN:UNKNOWN. Files stay world-readable so `python3` still runs by hand -- but the
    # harbor verifier, which chowns what it uploads into the container, then fails to write
    # reward.json and every trial dies with RewardFileNotFoundError.
    && tar -xzf /tmp/node.tar.gz -C /usr/local --strip-components=1 --no-same-owner \
    && rm -f /tmp/node.tar.gz \
    && npm i -g "opencode-ai@${OPENCODE_VERSION}" \
    && npm cache clean --force \
    && node --version && opencode --version

ARG AGENT_UID=10001
RUN useradd --create-home --uid "${AGENT_UID}" --shell /bin/bash agent     && mkdir -p /workspace     && chown -R agent:agent /workspace \
    && mkdir -p /home/agent/.nvm && : > /home/agent/.nvm/nvm.sh \
    && chown -R agent:agent /home/agent/.nvm

WORKDIR /workspace
USER agent
CMD ["/bin/bash"]
