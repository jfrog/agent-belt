# Task Tracker

A minimal CLI task management tool. Stores tasks as JSON on disk.

## Usage

```bash
tasktracker add "Buy groceries" --project personal
tasktracker list
tasktracker list --project personal
tasktracker done 1
tasktracker list --format json
```

## Structure

- `src/tasktracker/cli.py` - CLI entry point (argparse)
- `src/tasktracker/models.py` - Task and Project data classes
- `src/tasktracker/storage.py` - JSON file persistence
- `src/tasktracker/formatters.py` - Table and JSON output formatting

## Evaluation Scenarios

This fixture has 7 scenarios at 4 difficulty levels in
`examples/scenarios/experience/tasktracker-claude/`:

```bash
# Read-only (L1) - agent finds bugs without editing
belt eval examples/scenarios/experience/tasktracker-claude --tags L1 --modes rules

# Editing (L2) - agent fixes a single bug in an isolated worktree
belt eval examples/scenarios/experience/tasktracker-claude --tags L2 --modes rules

# All levels
belt eval examples/scenarios/experience/tasktracker-claude --modes rules
```

See [examples/README.md](../../README.md) for prerequisites and full usage.

## Known Issues

This project has intentional bugs for evaluation:

- `storage.py` - no file locking (concurrent access corruption)
- `cli.py` - missing `--format json` flag
- `formatters.py` - off-by-one in table column widths
- `models.py` - no `completed_at` timestamp field
- `storage.py` - no `delete()` method
