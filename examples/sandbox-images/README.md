# Reference sandbox images

Build container images for `--sandbox docker`. The framework does **not**
ship a registry of per-agent images: the install matrix for third-party
agent CLIs changes too often to track centrally, and tying image identity
to agent identity would couple the runner to every supported agent. Bring
your own image (BYOI), starting from the generic base.

## 1. Files in this directory

| File | Purpose |
|------|---------|
| `Dockerfile.generic` | Minimal base: `python:3.12-slim` + node 22 + git + ca-certificates. No agent CLIs preinstalled. Hardening tolerated by design. |
| `Dockerfile.cursor.example` | Worked derivation: `agent-belt-sandbox-generic` + `cursor-agent` install line. Illustrative; users ship their own. |
| `README.md` | This file. |

These files are **not bundled in the wheel** (`pyproject.toml` only
force-includes `examples/scenarios/showcase`). They live in the source
repository so users have a starting point; production users build their
own and host them in their own registry.

## 2. Build the generic base

```bash
docker build -t agent-belt-sandbox-generic:dev \
    -f examples/sandbox-images/Dockerfile.generic .
```

Use it from any scenario group's `_config.json`:

```json
{
  "agent": "claude-code",
  "sandbox": {
    "provider": "docker",
    "image": "agent-belt-sandbox-generic:dev",
    "env_passthrough": ["ANTHROPIC_API_KEY"]
  }
}
```

The agent CLI must already be on `PATH` inside the container -- the generic
base does **not** install any agent. Either derive an agent-specific image
(see step 3) or mount an agent install via `writable_paths`.

## 3. Derive an agent-specific image

Pattern: `FROM agent-belt-sandbox-generic:dev`, install the agent CLI in
one root layer, drop back to the unprivileged `belt` user, set
`WORKDIR /work`. Do **not** add an `ENTRYPOINT` -- the runner invokes the
agent CLI as the container command and an entrypoint would interfere with
that argv.

```bash
docker build -t agent-belt-sandbox-generic:dev \
    -f examples/sandbox-images/Dockerfile.generic .
docker build -t agent-belt-sandbox-cursor:dev \
    -f examples/sandbox-images/Dockerfile.cursor.example .
```

Use it:

```json
{
  "agent": "cursor",
  "sandbox": {
    "provider": "docker",
    "image": "agent-belt-sandbox-cursor:dev",
    "env_passthrough": ["CURSOR_API_KEY"]
  }
}
```

## 4. Hardening flags applied by the runner

The runner adds these flags at `docker run` time. Your image must tolerate
them (the generic base is built to do so):

| Flag | Effect |
|------|--------|
| `--cap-drop=ALL` | Container starts with zero Linux capabilities. |
| `--security-opt=no-new-privileges` | Setuid binaries cannot escalate. |
| `--read-only` | Root filesystem is immutable. |
| `--workdir=/work` | The host worktree is bind-mounted here, writable. |
| `-v $WORKTREE:/work:rw` | The only writable host path. |
| `--tmpfs=/home/agent:rw,size=256m,mode=1777` | Ephemeral, in-memory `$HOME` for agent caches; never reaches the host. |
| `-e HOME=/home/agent` | Pins `$HOME` to the tmpfs above. |
| `-v $PATH:$PATH:rw` | Each entry in `sandbox.writable_paths` (use sparingly). |
| `-e NAME ...` | Passthrough by exact name; values stay in the invoker's env. |
| `--add-host HOST:<resolved-ip>` | One per `sandbox.allowed_hosts` entry; the host name is resolved on the host (where DNS works) and the real IP is injected so the in-container resolver does not need an outbound DNS query. Skipped when `network_policy: "none"`. |
| `--network=none` | Added when `sandbox.network_policy: "none"` (see below). |

### 4.1. Network policy

`SandboxProfile.network_policy` controls outbound network at the kernel level:

| Value | Effect | When to use |
|-------|--------|-------------|
| `"open"` (default) | Docker's default bridge network: full outbound. `allowed_hosts` adds best-effort `--add-host` hints (DNS-only, NOT a firewall). | Any scenario whose agent calls an LLM/HTTP API. |
| `"none"` | `--network=none`: container has no network interfaces other than loopback. Outbound TCP / UDP / DNS all fail with `EHOSTUNREACH` or `ENETUNREACH` at the kernel level. `allowed_hosts` is silently dropped (no interface to register entries against). Loopback (`127.0.0.1`) still works for in-process IPC. | Offline scenarios: local code edits against a fixture, file-only tasks, stage where the agent has already been provisioned with prompts and must work without network. Strongest exfiltration guarantee belt currently offers. |

```json
{
  "agent": "claude-code",
  "sandbox": {
    "provider": "docker",
    "image": "agent-belt-sandbox-generic:dev",
    "network_policy": "none"
  }
}
```

Hostname-level allowlisting on top of `network_policy: "open"` (the
"allow `api.openai.com` only, deny everything else" case) is tracked
as future work; it requires a privileged sidecar with iptables rules
that belt cannot ship safely without an explicit operator opt-in.

`network_policy: "none"` is rejected at scenario start when the
chosen provider cannot enforce it. The host provider has no
isolation layer (the agent runs on the host kernel with the host
network), so a profile that pairs `provider: "host"` with
`network_policy: "none"` aborts the scenario with a single typed
error rather than silently running on the host's open network.
Switch to `provider: "docker"` to get the kernel-enforced policy,
or remove `network_policy` from the profile if the host network is
acceptable.

See [`docs/glossary/SANDBOXING.md`](../../docs/glossary/SANDBOXING.md) for the
threat model and the BYOI workflow in detail.

## 5. Why the framework does not ship per-agent images

- **Install matrices change.** Every agent CLI installs differently
  (Anthropic's `claude-code` via npm, `cursor-agent` via curl-bash, codex
  needs Node 22+, copilot needs `gh`, ...). Pinning a Dockerfile per agent
  would force the framework to track every install change.
- **Image identity should not equal agent identity.** A scenario can run
  multiple agents through the same image; an agent can be tested under
  multiple images. Decoupling keeps that flexibility.
- **Trust boundary.** Users hosting a sandbox image in their own registry
  control what is in it. The framework only ships the recipe.
