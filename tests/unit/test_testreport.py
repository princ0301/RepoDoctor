import pytest

from repodoctor.tools.testreport import fixed_tests, new_failures, parse_junit

SAMPLE = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="1" failures="1" skipped="1" tests="4">
<testcase classname="tests.test_a" name="test_ok" time="0.01"/>
<testcase classname="tests.test_a" name="test_bad" time="0.01">
  <failure message="assert 1 == 2">long traceback here</failure></testcase>
<testcase classname="tests.test_b" name="test_setup" time="0">
  <error message="AttributeError: boom">traceback</error></testcase>
<testcase classname="tests.test_b" name="test_skip" time="0">
  <skipped type="pytest.skip" message="nope"/></testcase>
</testsuite></testsuites>"""


def test_counts():
    r = parse_junit(SAMPLE)
    assert (r.total, r.passed, r.failed, r.errors, r.skipped) == (4, 1, 1, 1, 1)
    assert not r.clean


def test_failing_ids_and_messages():
    r = parse_junit(SAMPLE)
    assert r.failing == {
        "tests.test_a::test_bad": "assert 1 == 2",
        "tests.test_b::test_setup": "AttributeError: boom",
    }


def test_clean_report():
    xml = '<testsuite><testcase classname="t" name="a"/></testsuite>'
    r = parse_junit(xml)
    assert r.clean and r.passed == 1


def test_invalid_xml_raises():
    with pytest.raises(ValueError):
        parse_junit("not xml")


def test_new_failures_ignores_baseline_failures():
    base = parse_junit(SAMPLE)
    after_xml = SAMPLE.replace(
        "</testsuite>",
        '<testcase classname="tests.test_c" name="test_new"><failure message="x"/></testcase>'
        "</testsuite>",
    )
    after = parse_junit(after_xml)
    assert list(new_failures(base, after)) == ["tests.test_c::test_new"]


def test_fixed_tests():
    base = parse_junit(SAMPLE)
    after = parse_junit(SAMPLE.replace('<failure message="assert 1 == 2">long traceback here</failure>', ""))
    assert fixed_tests(base, after) == ["tests.test_a::test_bad"]