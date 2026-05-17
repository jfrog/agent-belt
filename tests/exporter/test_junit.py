# (c) JFrog Ltd. (2026)

"""``junit`` exporter behaviour: structure, escaping, trial collapse."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from belt.exporter.entities import ExportContext
from belt.exporter.junit import JUnitExporter


def _parse(out: Path) -> ET.Element:
    tree = ET.parse(out)
    return tree.getroot()


class TestStructure:
    def test_root_is_testsuites(self, export_context: ExportContext, tmp_path: Path):
        out = tmp_path / "report.xml"
        JUnitExporter().export(export_context, out, {})
        root = _parse(out)
        assert root.tag == "testsuites"
        # 2 scenarios, both in the same group g1.
        assert int(root.attrib["tests"]) == 2
        assert int(root.attrib["failures"]) == 1
        assert root.attrib["name"]  # default: tmp_path basename

    def test_one_testsuite_per_group(self, export_context: ExportContext, tmp_path: Path):
        out = tmp_path / "report.xml"
        JUnitExporter().export(export_context, out, {})
        root = _parse(out)
        suites = list(root.findall("testsuite"))
        assert len(suites) == 1
        assert suites[0].attrib["name"] == "g1"
        assert int(suites[0].attrib["tests"]) == 2

    def test_failure_carries_message_and_body(self, export_context: ExportContext, tmp_path: Path):
        out = tmp_path / "report.xml"
        JUnitExporter().export(export_context, out, {})
        root = _parse(out)
        beta = root.find(".//testcase[@name='beta']")
        assert beta is not None
        failure = beta.find("failure")
        assert failure is not None
        assert "rule check" in failure.attrib["message"].lower() or "low" in failure.attrib["message"].lower()
        assert "execution/exited_zero" in (failure.text or "")


class TestEscaping:
    def test_xml_special_chars_escaped(self, export_context: ExportContext, tmp_path: Path):
        out = tmp_path / "report.xml"
        JUnitExporter().export(export_context, out, {})
        # The failing fixture embeds <script>alert(1)</script> in LLM reasoning.
        # Re-parsing must succeed and the angle brackets must NOT survive raw.
        raw = out.read_text()
        assert "<script>" not in raw
        assert "&lt;script&gt;" in raw
        # Tree parses cleanly = escaping is correct for XML.
        ET.parse(out)


class TestTrialSemantics:
    def test_one_testcase_per_base_scenario(self, trial_export_context: ExportContext, tmp_path: Path):
        out = tmp_path / "report.xml"
        JUnitExporter().export(trial_export_context, out, {})
        root = _parse(out)
        cases = list(root.findall(".//testcase"))
        # Two trials of "alpha" collapse to one testcase.
        assert len(cases) == 1
        assert cases[0].attrib["name"] == "alpha"

    def test_trial_properties_emitted(self, trial_export_context: ExportContext, tmp_path: Path):
        out = tmp_path / "report.xml"
        JUnitExporter().export(trial_export_context, out, {})
        root = _parse(out)
        props = {p.attrib["name"]: p.attrib["value"] for p in root.findall(".//property")}
        assert props.get("trials") == "2"
        assert props.get("trials_passed") == "1"
        assert props.get("pass_at_1", "").startswith("0.5")
        assert props.get("pass_at_3", "").startswith("0.875")
        assert props.get("pass_pow_3", "").startswith("0.125")


class TestSuiteName:
    def test_custom_suite_name_via_options(self, export_context: ExportContext, tmp_path: Path):
        out = tmp_path / "report.xml"
        JUnitExporter().export(export_context, out, {"suite_name": "matrix-1"})
        root = _parse(out)
        assert root.attrib["name"] == "matrix-1"
