import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

_MSG_MAX = 300

@dataclass
class SuiteReport:
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    failing: dict[str, str] = field(default_factory=dict)  # test id -> short message

    @property
    def clean(self) -> bool:
        return not self.failing

def parse_junit(xml_text: str) -> SuiteReport:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise ValueError(f"invalid JUnit XML: {e}") from e

    report = SuiteReport()
    for case in root.iter("testcase"):
        report.total += 1
        cls, name = case.get("classname"), case.get("name", "")
        test_id = f"{cls}::{name}" if cls else name

        failure = case.find("failure")
        error = case.find("error")
        if failure is not None or error is not None:
            node = failure if failure is not None else error
            if failure is not None:
                report.failed += 1
            else:
                report.errors += 1
            lines = (node.get("message") or node.text or "").strip().splitlines()
            report.failing[test_id] = lines[0][:_MSG_MAX] if lines else ""
        elif case.find("skipped") is not None:
            report.skipped += 1
        else:
            report.passed += 1
    return report


def new_failures(baseline: SuiteReport, after: SuiteReport) -> dict[str, str]:
    """Tests failing now that were not failing in the baseline."""
    return {k: v for k, v in after.failing.items() if k not in baseline.failing}


def fixed_tests(baseline: SuiteReport, after: SuiteReport) -> list[str]:
    """Tests that failed in the baseline and pass now."""
    return sorted(set(baseline.failing) - set(after.failing))

def parse_outcomes(xml_text: str) -> dict[str, str]:
    """Test id -> 'passed' | 'failed' | 'error' | 'skipped'.

    The worst outcome wins when pytest reports one test several times
    (for example a failure in the call plus an error in teardown)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise ValueError(f"invalid JUnit XML: {e}") from e

    rank = {"passed": 0, "skipped": 1, "failed": 2, "error": 3}
    out: dict[str, str] = {}
    for case in root.iter("testcase"):
        cls, name = case.get("classname"), case.get("name", "")
        test_id = f"{cls}::{name}" if cls else name
        if case.find("error") is not None:
            status = "error"
        elif case.find("failure") is not None:
            status = "failed"
        elif case.find("skipped") is not None:
            status = "skipped"
        else:
            status = "passed"
        prev = out.get(test_id)
        if prev is None or rank[status] > rank[prev]:
            out[test_id] = status
    return out