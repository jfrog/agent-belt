# (c) JFrog Ltd. (2026)

"""Data models for tasks and projects."""

import time
from dataclasses import dataclass, field


@dataclass
class Task:
    id: int
    title: str
    project: str = "default"
    done: bool = False
    created_at: float = field(default_factory=time.time)
    # BUG: missing completed_at timestamp - done tasks have no completion time

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "project": self.project,
            "done": self.done,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            title=data["title"],
            project=data.get("project", "default"),
            done=data.get("done", False),
            created_at=data.get("created_at", 0),
        )


@dataclass
class Project:
    name: str
    tasks: list = field(default_factory=list)

    def active_tasks(self):
        return [t for t in self.tasks if not t.done]

    def completed_tasks(self):
        return [t for t in self.tasks if t.done]
