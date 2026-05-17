# (c) JFrog Ltd. (2026)

"""JSON file storage for tasks."""

import json
import os

DEFAULT_PATH = os.path.expanduser("~/.tasktracker.json")


def load_tasks(path=DEFAULT_PATH):
    # BUG: no file locking - concurrent reads/writes can corrupt data
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    from tasktracker.models import Task

    return [Task.from_dict(d) for d in data]


def save_tasks(tasks, path=DEFAULT_PATH):
    # BUG: no file locking - concurrent writes can corrupt or lose data
    data = [t.to_dict() for t in tasks]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def next_id(tasks):
    if not tasks:
        return 1
    return max(t.id for t in tasks) + 1
