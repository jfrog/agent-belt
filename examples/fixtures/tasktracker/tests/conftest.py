# (c) JFrog Ltd. (2026)

"""Shared test fixtures."""

import json

import pytest


@pytest.fixture
def tmp_storage(tmp_path):
    """Provide a temporary file path for task storage."""
    return str(tmp_path / "tasks.json")


@pytest.fixture
def sample_tasks_file(tmp_path):
    """Create a temp file pre-populated with sample tasks."""
    path = str(tmp_path / "tasks.json")
    data = [
        {"id": 1, "title": "Buy groceries", "project": "personal", "done": False, "created_at": 1000.0},
        {"id": 2, "title": "Write tests", "project": "work", "done": False, "created_at": 1001.0},
        {"id": 3, "title": "Old task", "project": "personal", "done": True, "created_at": 999.0},
    ]
    with open(path, "w") as f:
        json.dump(data, f)
    return path
