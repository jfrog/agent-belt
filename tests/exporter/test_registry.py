# (c) JFrog Ltd. (2026)

"""Registry behaviour: built-ins, default-deny gate, dotted-import escape hatch."""

from __future__ import annotations

import pytest

from belt import envvars
from belt.errors import ConfigError
from belt.exporter.base import BaseExporter
from belt.exporter.registry import available_exporters, get_exporter_class


class TestBuiltIns:
    def test_all_four_builtins_registered(self):
        names = available_exporters()
        for expected in ("csv", "jsonl", "junit", "markdown"):
            assert expected in names, f"built-in '{expected}' missing from {names}"

    @pytest.mark.parametrize("name", ["csv", "jsonl", "junit", "markdown"])
    def test_built_in_resolves_to_baseexporter_subclass(self, name: str):
        cls = get_exporter_class(name)
        assert issubclass(cls, BaseExporter)
        instance = cls()
        # Each built-in declares ``name`` matching its registry key, mirroring
        # the BaseScorer.name convention.
        assert instance.name == name


class TestDefaultDeny:
    def test_unknown_name_without_gate_raises(self, monkeypatch: pytest.MonkeyPatch):
        # Make sure no shell-leaked toggle relaxes the gate.
        monkeypatch.delenv(envvars.ALLOW_ARBITRARY_EXPORTER, raising=False)
        with pytest.raises(ConfigError) as excinfo:
            get_exporter_class("does_not_exist")
        assert "--allow-arbitrary-exporter" in str(excinfo.value)
        assert envvars.ALLOW_ARBITRARY_EXPORTER in str(excinfo.value)

    def test_dotted_path_without_gate_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv(envvars.ALLOW_ARBITRARY_EXPORTER, raising=False)
        with pytest.raises(ConfigError):
            get_exporter_class("belt.exporter.csv:CsvExporter")

    def test_gate_unlocks_dotted_path(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(envvars.ALLOW_ARBITRARY_EXPORTER, "1")
        cls = get_exporter_class("belt.exporter.csv:CsvExporter")
        assert issubclass(cls, BaseExporter)

    def test_non_baseexporter_target_rejected(self, monkeypatch: pytest.MonkeyPatch):
        # ``str`` is a class but not a BaseExporter subclass; the registry
        # must reject it instead of returning a bogus exporter type.
        monkeypatch.setenv(envvars.ALLOW_ARBITRARY_EXPORTER, "1")
        with pytest.raises(ConfigError) as excinfo:
            get_exporter_class("builtins:str")
        assert "BaseExporter" in str(excinfo.value)
