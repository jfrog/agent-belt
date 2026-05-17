# (c) JFrog Ltd. (2026)

"""CLI subcommand entry points.

Each module here is a thin argparse + ``main()`` driver for one ``belt``
subcommand. Business logic lives in phase libraries (``belt.runner``,
``belt.scorer``, ``belt.aggregator``) and standalone modules.

The top-level dispatcher at ``belt.cli`` imports the ``main`` callable
from each command module here.

Note: these modules are internal - not part of the public API. The public
surface is the ``belt`` console script and its subcommands. Forks
should call ``belt`` directly rather than importing from this package.
"""
