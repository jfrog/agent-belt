# (c) JFrog Ltd. (2026)

"""Parity contract for built-in agents.

Behavioral invariants asserted across every agent in ``_AGENT_REGISTRY``:

1. ``config.workspace_dir`` is propagated to ``subprocess.Popen(cwd=...)``.
   Without this, editing scenarios that rely on workspace isolation (e.g.
   git worktrees) silently run in the harness cwd and produce diffs against
   the wrong tree - a class of bug that is invisible until a workspace-aware
   scenario actually runs.

2. ``supported_output_fields()`` only declares fields from the canonical
   ``AGENT_SPECIFIC_FIELDS`` set defined in ``agent/base.py``. A
   misspelled field name silently disables every scenario expectation that
   references it.

3. ``denied_flags()`` returns at least one flag for every concrete agent.
   Empty ``denied_flags`` means a scenario can inject any CLI flag the agent
   accepts - including capability-broadening ones like ``--with-extension``
   (Goose) or ``--remote`` (Copilot). The base-class default IS the unsafe
   default; concrete adapters must enumerate which flags they refuse.

What this test deliberately does NOT enforce
--------------------------------------------
Surface declarations (``CREDENTIAL_ENV``, ``parse_stream_event`` overrides,
``cli_options``, ``scoring_strategy``, ``health_check``) are intentionally
not gated. Some agents legitimately inherit framework defaults - e.g.
multi-provider agents cannot pin a single ``CREDENTIAL_ENV``, and NDJSON
agents that use the generic event renderer have nothing to override in
``parse_stream_event``. Forcing empty overrides would be busywork without
behavioral protection.
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from belt.agent.base import AGENT_SPECIFIC_FIELDS
from belt.agent.registry import _AGENT_REGISTRY, get_agent_class
from belt.runner.entities import AgentConfig
from belt.scenario import GroupConfig

_SENTINEL_WORKSPACE = "/tmp/belt-parity-sentinel"


@pytest.mark.parametrize("agent_name", sorted(_AGENT_REGISTRY))
class TestAgentParity:
    """Parametrized parity assertions over every built-in agent."""

    def test_workspace_dir_propagated_to_subprocess_cwd(self, agent_name: str) -> None:
        """Every agent must invoke its subprocess with ``cwd=config.workspace_dir``.

        Catches the class of bug where an agent silently runs in the harness
        cwd instead of the isolated workspace (git worktree), making editing
        scenarios produce diffs against the wrong tree.
        """
        cls = get_agent_class(agent_name)
        module = __import__(cls.__module__, fromlist=["subprocess"])
        if not hasattr(module, "subprocess"):
            pytest.skip(
                f"{agent_name}: agent does not import `subprocess` at module scope - "
                f"cwd propagation cannot be inspected by this test"
            )

        captured: list[dict] = []

        def fake_popen(cmd, *_a, **kwargs):
            captured.append(dict(kwargs))
            mock = MagicMock()
            mock.stdout = StringIO("")
            mock.stderr = StringIO("")
            mock.returncode = 0
            mock.wait = MagicMock()
            mock.pid = 12345
            return mock

        with patch.object(module.subprocess, "Popen", side_effect=fake_popen):
            agent = cls()
            agent.setup(
                AgentConfig(
                    group_config=GroupConfig(agent=agent_name),
                    scenario_name="parity-test",
                    workspace_dir=_SENTINEL_WORKSPACE,
                )
            )
            try:
                agent.execute("ping", [])
            except Exception as e:
                if not captured:
                    pytest.skip(f"{agent_name}: execute() failed before Popen was called: {e}")

        if not captured:
            pytest.skip(f"{agent_name}: agent did not invoke subprocess.Popen on execute()")

        cwd = captured[0].get("cwd")
        assert cwd == _SENTINEL_WORKSPACE, (
            f"{agent_name}: subprocess.Popen called with cwd={cwd!r}, "
            f"expected {_SENTINEL_WORKSPACE!r} (config.workspace_dir). "
            f"The agent is silently ignoring workspace isolation - editing "
            f"scenarios will produce diffs against the wrong tree. Pass "
            f"cwd=self._workspace_dir to subprocess.Popen in _execute_streaming()."
        )

    def test_supported_output_fields_subset_of_known(self, agent_name: str) -> None:
        """``supported_output_fields()`` may only declare fields from the canonical set.

        Misspelled or invented field names silently disable scenario expectations
        that reference them. The canonical set lives in ``agent/base.py``.
        """
        cls = get_agent_class(agent_name)
        fields = cls.supported_output_fields()
        unknown = fields - AGENT_SPECIFIC_FIELDS
        assert not unknown, (
            f"{agent_name}: supported_output_fields() declares unknown field(s): "
            f"{sorted(unknown)}. Allowed: {sorted(AGENT_SPECIFIC_FIELDS)}"
        )

    def test_denied_flags_non_empty(self, agent_name: str) -> None:
        """Every concrete agent must enumerate at least one denied CLI flag.

        ``BaseAgentAdapter.denied_flags`` returns ``frozenset()`` by
        default - the unsafe default. Concrete adapters must override
        with the published capability-broadening flags their CLI
        exposes (e.g. ``--yolo``, ``--with-extension``,
        ``--dangerously-skip-permissions``, ``--remote``). An adapter
        that inherits the empty default lets scenarios silently bypass
        every safety surface the underlying CLI offers.
        """
        cls = get_agent_class(agent_name)
        denied = cls.denied_flags()
        assert isinstance(
            denied, frozenset
        ), f"{agent_name}: denied_flags() returned {type(denied).__name__}, expected frozenset"
        assert len(denied) > 0, (
            f"{agent_name}: denied_flags() is empty. Override the base-class "
            f"default to enumerate the capability-broadening flags this CLI "
            f"accepts (search the CLI --help for ``--yolo``, ``--with-*``, "
            f"``--dangerously-*``, ``--remote``, ``--allow-all*`` or equivalents)."
        )
        for flag in denied:
            assert flag.startswith("-"), (
                f"{agent_name}: denied_flags() entry {flag!r} does not start with "
                f"'-'; only CLI flag tokens belong here."
            )

    def test_constructor_rejects_unknown_kwargs(self, agent_name: str) -> None:
        """Built-in agent constructors must not silently swallow unknown kwargs.

        ``**kwargs`` in ``__init__`` makes the constructor accept arbitrary args
        the runner thinks it validated, defeating the single-source-of-truth
        kwarg validation in ``runner.context.create_agent``. Catches future
        regressions of the ``**_kwargs: Any`` pattern.
        """
        cls = get_agent_class(agent_name)
        with pytest.raises(TypeError):
            cls(definitely_not_a_real_option=1)  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        ("fixture_dirname", "expected_token"),
        [
            ("auth_failure", "authentication_failed"),
            ("rate_limited", "rate_limited"),
            ("timeout", "timeout"),
            ("refused", "refused"),
        ],
    )
    def test_classifies_error_type(self, agent_name: str, fixture_dirname: str, expected_token: str) -> None:
        """Every adapter must classify a fixture for every canonical error type it can emit.

        The taxonomy in :mod:`belt.entities` enumerates four
        emit-able tokens: ``authentication_failed``, ``rate_limited``,
        ``timeout``, ``refused``. (``unknown`` is the fallback when
        ``has_error=true`` but no signal classified, so it has no
        per-agent fixture.) Each fixture lives at
        ``tests/agent/fixtures/<dirname>/<agent-name>.ndjson`` and is
        the smallest input that triggers ``has_error=true`` plus
        carries a recognisable substring for that error type. Adapters
        adding or changing their failure-classification logic must
        keep these fixtures in lockstep.

        A missing fixture is a hard fail - the test does not skip - so
        a newly registered adapter cannot ship without coverage for
        every canonical type, and the cross-agent parity story stays
        honest.
        """
        from pathlib import Path

        cls = get_agent_class(agent_name)
        fixtures_dir = Path(__file__).parent / "fixtures" / fixture_dirname
        candidates = list(fixtures_dir.glob(f"{agent_name}.*"))
        assert candidates, (
            f"{agent_name}: no {fixture_dirname!r} fixture found under "
            f"{fixtures_dir} (expected '{agent_name}.ndjson' or similar). "
            f"Add a fixture and update the parity test - see "
            f"tests/agent/fixtures/README.md."
        )
        fixture_text = candidates[0].read_text()

        agent = cls()
        # ``setup`` is required by some adapters before ``fetch_results``
        # can mutate per-instance state (session_id caches, etc.).
        agent.setup(
            AgentConfig(
                group_config=GroupConfig(agent=agent_name),
                scenario_name=f"{fixture_dirname}-parity",
            )
        )
        to = agent.fetch_results(fixture_text)
        assert to.has_error is True, (
            f"{agent_name}/{fixture_dirname}: fixture did not trigger "
            f"has_error=true. Either update the adapter's has_error "
            f"logic or strengthen the fixture to include the structured "
            f"signal it needs."
        )
        assert to.error_type == expected_token, (
            f"{agent_name}/{fixture_dirname}: classified fixture as "
            f"{to.error_type!r}, expected {expected_token!r}. The adapter "
            f"must wire classify_error (or its own structured-error parsing) "
            f"into fetch_results so {expected_token!r} failures get a "
            f"stable label."
        )

    def test_cli_options_match_constructor_signature(self, agent_name: str) -> None:
        """Every declared ``cli_option`` must be a kwarg the ``__init__`` accepts.

        ``runner.context.create_agent`` resolves env-var fallbacks from
        ``cli_options()`` and forwards every resolved key as a kwarg to the
        constructor. A declared option whose name is not on ``__init__`` causes
        ``create_agent`` to crash with ``AgentArgError`` whenever the
        corresponding env var is set in the user's environment - exactly the
        failure mode that the parameterless hardening was meant to avoid.
        This contract test prevents regression of either side: adding an
        option without plumbing it, or removing a kwarg without dropping the
        option declaration.
        """
        import inspect

        cls = get_agent_class(agent_name)
        options = cls.cli_options()
        if not options:
            return  # nothing to verify

        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        accepts_var_keyword = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        for opt in options:
            if accepts_var_keyword:
                continue  # ``**kwargs`` constructors accept anything (codex/goose/opencode)
            assert opt.name in params, (
                f"{agent_name}: cli_options() declares {opt.name!r} but "
                f"__init__ does not accept it. Either add ``{opt.name}: T = None`` "
                f"to __init__ (and plumb it through execute), or drop the option "
                f"from cli_options(). Declaring an option without a kwarg makes "
                f"create_agent() crash with AgentArgError when the option's "
                f"env_var is set in the user's environment."
            )
