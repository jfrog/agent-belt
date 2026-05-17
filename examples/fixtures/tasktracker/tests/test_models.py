# (c) JFrog Ltd. (2026)

"""Tests for task models."""

from tasktracker.models import Project, Task


class TestTask:
    def test_create_task(self):
        task = Task(id=1, title="Test task")
        assert task.id == 1
        assert task.title == "Test task"
        assert task.project == "default"
        assert task.done is False

    def test_to_dict(self):
        task = Task(id=1, title="Test", project="work", done=True, created_at=1000.0)
        d = task.to_dict()
        assert d["id"] == 1
        assert d["title"] == "Test"
        assert d["project"] == "work"
        assert d["done"] is True

    def test_from_dict(self):
        data = {"id": 2, "title": "From dict", "project": "home", "done": False, "created_at": 500.0}
        task = Task.from_dict(data)
        assert task.id == 2
        assert task.title == "From dict"


class TestProject:
    def test_active_tasks(self):
        t1 = Task(id=1, title="Active", done=False)
        t2 = Task(id=2, title="Done", done=True)
        proj = Project(name="test", tasks=[t1, t2])
        assert len(proj.active_tasks()) == 1
        assert proj.active_tasks()[0].id == 1

    def test_completed_tasks(self):
        t1 = Task(id=1, title="Active", done=False)
        t2 = Task(id=2, title="Done", done=True)
        proj = Project(name="test", tasks=[t1, t2])
        assert len(proj.completed_tasks()) == 1
        assert proj.completed_tasks()[0].id == 2
