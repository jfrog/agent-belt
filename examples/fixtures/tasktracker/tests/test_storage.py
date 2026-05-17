# (c) JFrog Ltd. (2026)

"""Tests for storage layer."""

from tasktracker.models import Task
from tasktracker.storage import load_tasks, next_id, save_tasks


class TestStorage:
    def test_load_empty(self, tmp_storage):
        tasks = load_tasks(tmp_storage)
        assert tasks == []

    def test_save_and_load(self, tmp_storage):
        tasks = [Task(id=1, title="Test task", created_at=1000.0)]
        save_tasks(tasks, tmp_storage)
        loaded = load_tasks(tmp_storage)
        assert len(loaded) == 1
        assert loaded[0].title == "Test task"

    def test_load_sample(self, sample_tasks_file):
        tasks = load_tasks(sample_tasks_file)
        assert len(tasks) == 3
        assert tasks[0].title == "Buy groceries"

    def test_next_id_empty(self):
        assert next_id([]) == 1

    def test_next_id(self):
        tasks = [Task(id=5, title="Five"), Task(id=3, title="Three")]
        assert next_id(tasks) == 6
