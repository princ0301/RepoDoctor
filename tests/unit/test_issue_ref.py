import pytest

from repodoctor.tools.issue_ref import (
    InvalidIssueUrl,
    IssueRef,
    RepoNotAllowed,
    check_allowed,
    parse_issue_url,
)


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/pallets/click/issues/123",
        "https://github.com/pallets/click/issues/123/",
        "https://www.github.com/pallets/click/issues/123",
        "https://github.com/pallets/click/issues/123#issuecomment-99",
        "https://github.com/pallets/click/issues/123?foo=bar",
        "  https://github.com/pallets/click/issues/123  ",
    ],
)
def test_parse_valid(url):
    assert parse_issue_url(url) == IssueRef("pallets", "click", 123)


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/pallets/click/pull/123",
        "https://gitlab.com/pallets/click/issues/123",
        "https://github.com/pallets/click/issues/",
        "https://github.com/pallets/click/issues/abc",
        "https://github.com/pallets/click/issues/123abc",
        "https://github.com/pallets",
        "not a url",
        "",
    ],
)
def test_parse_invalid(url):
    with pytest.raises(InvalidIssueUrl):
        parse_issue_url(url)


def test_allowlist_is_case_insensitive():
    check_allowed(IssueRef("Pallets", "Click", 1), ["pallets/click"])


def test_allowlist_rejects_unknown_repo():
    with pytest.raises(RepoNotAllowed):
        check_allowed(IssueRef("a", "b", 1), ["pallets/click"])


def test_empty_allowlist_rejects_everything():
    with pytest.raises(RepoNotAllowed):
        check_allowed(IssueRef("a", "b", 1), [])