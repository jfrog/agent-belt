# (c) JFrog Ltd. (2026)

"""CLI entry point for tasktracker."""

import argparse
import sys

from tasktracker.formatters import format_table
from tasktracker.models import Task
from tasktracker.storage import load_tasks, next_id, save_tasks


def build_parser():
    parser = argparse.ArgumentParser(description="Task Tracker CLI")
    subparsers = parser.add_subparsers(dest="command")

    add_parser = subparsers.add_parser("add", help="Add a new task")
    add_parser.add_argument("title", help="Task title")
    add_parser.add_argument("--project", default="default", help="Project name")

    list_parser = subparsers.add_parser("list", help="List tasks")
    list_parser.add_argument("--project", default=None, help="Filter by project")
    # BUG: missing --format flag - no way to get JSON output from CLI

    done_parser = subparsers.add_parser("done", help="Mark a task as done")
    done_parser.add_argument("task_id", type=int, help="Task ID to mark done")

    return parser


def cmd_add(args):
    tasks = load_tasks()
    task = Task(id=next_id(tasks), title=args.title, project=args.project)
    tasks.append(task)
    save_tasks(tasks)
    print(f"Added task {task.id}: {task.title}")


def cmd_list(args):
    tasks = load_tasks()
    if args.project:
        tasks = [t for t in tasks if t.project == args.project]
    print(format_table(tasks))


def cmd_done(args):
    tasks = load_tasks()
    for task in tasks:
        if task.id == args.task_id:
            task.done = True
            save_tasks(tasks)
            print(f"Task {task.id} marked as done.")
            return
    print(f"Task {args.task_id} not found.", file=sys.stderr)
    sys.exit(1)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "add": cmd_add,
        "list": cmd_list,
        "done": cmd_done,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
