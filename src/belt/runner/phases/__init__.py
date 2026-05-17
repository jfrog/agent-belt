# (c) JFrog Ltd. (2026)

"""Run-phase pipeline modules.

Each module here implements one phase of ``belt run``:

- ``parse_filter`` - discover groups, apply ``--scenarios`` / ``--tags`` filters
- ``setup_groups`` - agent setup in parallel + orphan cleanup
- ``run_scenarios`` - scenario dispatch (scenario-level parallelism)
- ``teardown`` - agent teardown + manifest unregister

Modules are imported by ``commands/run.py``; they take a ``RunContext``
and return either an exit code or extend the context's mutable fields.
"""
