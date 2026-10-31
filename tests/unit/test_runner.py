import pytest

from repodoctor.runner import parse_issue_file


def test_title_and_body():
    assert parse_issue_file("# Bug in x\n\nIt breaks.\nMore.\n") == (
        "Bug in x", "It breaks.\nMore.",
    )


def test_title_only():
    assert parse_issue_file("Just a title") == ("Just a title", "")


def test_empty_file_is_rejected():
    with pytest.raises(ValueError):
        parse_issue_file("  \n ")