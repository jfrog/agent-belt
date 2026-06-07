# Sandboxing

How `belt` isolates an agent subprocess, what each provider enforces,
and where to read the code.

## 1. Providers

Selected via `--sandbox NAME`, `BELT_SANDBOX_PROVIDER`, or per-group
`sandbox.provider` in `_config.json`. Resolution order is documented in
[CONFIGURATION.md](CONFIGURATION.md). `belt doctor` lists what is
available on the current host.

| Provider | Code | Isolation |
|---|---|---|
| `host` | [`src/belt/runner/sandbox/host.py`](../../src/belt/runner/sandbox/host.py) | None. The agent runs as the invoking user with the user's `$PATH`, full filesystem access, and unrestricted network. |
| `docker` | [`src/belt/runner/sandbox/docker.py`](../../src/belt/runner/sandbox/docker.py) | Containerised. The kernel-enforced invariants are listed in §2. |
| Third-party | `belt.sandbox_providers` entry-point group; see [PLUGGABILITY.md §10](PLUGGABILITY.md#10-authoring-a-sandbox-provider) | Provider-defined. Validation against `SandboxProfile` is the provider's responsibility. |

## 2. Kernel-enforced invariants under `--sandbox docker`

Every row is enforced by the host kernel through the `docker` CLI. The
test cited in the right column exercises the row against a live
container; the `skipif` guard auto-skips when docker is not installed.

| Invariant | Implementation | Test |
|---|---|---|
| Capability drop | `--cap-drop=ALL`, `--security-opt=no-new-privileges` | `tests/runner/sandbox/test_docker_e2e.py::test_e2e_capability_dropped_mount_blocked` |
| Read-only rootfs | `--read-only` (with tmpfs `$HOME` for transient writes) | `test_e2e_rootfs_is_readonly_writes_to_etc_fail` |
| Filesystem scope | Bind-mounts the per-scenario worktree and the scenario-declared `writable_paths` only | `test_e2e_writes_outside_worktree_and_extras_fail` |
| Non-root user | Sandbox image declares a non-root `USER` (reference image: `USER belt`, uid 1000); the runner never overrides it back to root | `test_e2e_runs_as_unprivileged_user_not_root` |
| Network isolation | `network_policy: "none"` → `--network=none` (outbound TCP and DNS fail at the kernel with `EHOSTUNREACH` / `ENETUNREACH`) | `test_e2e_network_policy_none_blocks_outbound_tcp_at_kernel_level`, `test_e2e_network_policy_none_blocks_dns_resolution` |
| Env passthrough | Exact-name allowlist (`SandboxProfile.env_passthrough` ∪ the agent's declared required env); no wildcards, no shell expansion | `test_e2e_env_not_in_passthrough_does_not_leak`, `test_e2e_env_passthrough_value_delivered_when_listed` |
| Dangerous-flag denylist | Per-agent `denied_flags()` blocks `--yolo`, `--dangerously-skip-permissions`, `--sandbox=danger-full-access`, `--allow-all*`, etc. Enforced before the sandbox layer, so it applies on every provider including `host`. | `tests/test_security.py::TestAgentDeniedFlagsDefaults` |

## 3. `network_policy`

`SandboxProfile.network_policy` is a closed `Literal["open", "none"]`.

- `"open"`: docker bridge network with `SandboxProfile.allowed_hosts`
  materialised as `--add-host` entries. DNS hint only; does not block
  other outbound traffic.
- `"none"`: `--network=none` removes every namespace interface except
  loopback. Outbound syscalls fail in the kernel.

## 4. SandboxProfile schema

`GroupConfig.sandbox` accepts a `SandboxProfile` per group; see
[CONFIGURATION.md](CONFIGURATION.md#3-environment-variables) and the
Pydantic model in [`src/belt/scenario.py`](../../src/belt/scenario.py).
Per-group fields take precedence over `BELT_SANDBOX_PROVIDER` and the
`--sandbox` flag for the fields they set.

## 5. Authoring a provider

See [PLUGGABILITY.md §10](PLUGGABILITY.md#10-authoring-a-sandbox-provider).
