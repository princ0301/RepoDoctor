from datetime import UTC, datetime

import httpx
import pytest

from repodoctor.tools.conventions import (
    analyze,
    changelog_style,
    checklist_tick,
    classify,
    convention_findings,
    draft_changelog_entry,
    fragment_path,
    homes_for_sources,
    load_or_mine,
    mine,
    new_test_findings,
    render_checklist,
    update_checkboxes,
)
from repodoctor.tools.github import github_client
from repodoctor.tools.repo import commit_files, head_sha

NOW = datetime(2026, 11, 20, tzinfo=UTC)
TEMPLATE = """<!--
Before opening a PR, open a ticket.
- fixes #<issue number>
-->

Checklist:

- [ ] Add tests that demonstrate the correct behavior of the change. Tests should fail without the change.
- [ ] Add or update relevant docs, in the docs folder and in code.
- [ ] Add an entry in `CHANGES.rst` summarizing the change and linking to the issue.
- [ ] Add `.. versionchanged::` entries in any relevant code docs.
- [ ] Run `pre-commit` hooks and fix any issues.
- [ ] Run `pytest` and `tox`, no tests failed.
"""
ALL_TRUE = {
    "tests_added": True, "test_fails_without_fix": True, "changelog_touched": True,
    "lint_passed": True, "suite_passed": True,
}


def rec(number, issues, files, new_tests=()):
    return {
        "number": number, "title": "t", "issues": issues, "merged_at": "x",
        "files": [
            {"path": p, "status": s, "kind": classify(p), "additions": 1, "deletions": 0}
            for p, s in files
        ],
        "new_tests": list(new_tests),
    }


def mock_client(routes, texts=None, counter=None):
    def handler(request):
        if counter is not None:
            counter.append(request.url.path)
        path = request.url.path
        if path in routes:
            return httpx.Response(200, json=routes[path])
        if texts and path in texts:
            return httpx.Response(200, text=texts[path])
        return httpx.Response(404, json={})

    return github_client("", transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("path, kind", [
    ("tests/test_x.py", "test"), ("pkg/mod.py", "source"), ("CHANGELOG", "changelog"),
    ("CHANGES.rst", "changelog"), ("docs/changes.rst", "changelog"),
    ("changelog/13835.bugfix.rst", "fragment"), ("newsfragments/12.feature", "fragment"),
    ("docs/index.rst", "docs"), ("setup.cfg", "other"),
])
def test_classify(path, kind):
    assert classify(path) == kind


def test_preticked_template_boxes_stay_ticked():
    out = render_checklist(
        "- [X] Allow maintainers to push\n- [ ] Add tests that fail without the change\n", {}
    )
    assert "- [x] Allow maintainers to push" in out and "- [ ] Add tests" in out


def test_suite_items_follow_the_suite_not_the_new_test():
    assert checklist_tick("Run pytest, no tests failed", {"suite_passed": True})
    assert not checklist_tick(
        "Run pytest, no tests failed",
        {"suite_passed": False, "tests_added": True, "test_fails_without_fix": True},
    )
    assert checklist_tick(
        "Include new tests or update existing tests when applicable",
        {"tests_added": True, "test_fails_without_fix": True},
    )


def test_history_from_the_same_folder_is_labelled_and_call_based_hints_are_added(make_git_repo):
    root = make_git_repo({
        "pkg/splitter.py": "x = 1\n", "pkg/grouping.py": "y = 1\n",
        "tests/test_grouping.py": "def test_g():\n    pass\n",
        "tests/test_split.py": "import lib\n\n\ndef test_s():\n    lib.split('a')\n",
    })
    base = head_sha(root)
    (root / "tests" / "test_issue_5.py").write_text(
        "import lib\n\n\ndef test_x():\n    lib.split('b; c')\n"
    )
    commit_files(root, ["tests/test_issue_5.py"], "add")
    recs = [rec(n, [n], [("pkg/grouping.py", "modified"), ("tests/test_grouping.py", "modified")])
            for n in (1, 2, 3)]
    conv = {"prs": recs, "stats": analyze(recs)}
    found = {f.code: f.message for f in new_test_findings(
        conv, root, base, ["pkg/splitter.py", "tests/test_issue_5.py"], 5)}
    assert "same folder" in found["new_test_file_folder"]
    assert "tests/test_grouping.py" in found["new_test_file_folder"]
    assert "tests/test_split.py" in found["new_test_file"]


def sample_records():
    return [
        rec(1, [10], [("pkg/a.py", "modified"), ("tests/test_regressions.py", "modified"),
                      ("CHANGELOG", "modified")], ["test_issue10_foo"]),
        rec(2, [11], [("pkg/a.py", "modified"), ("tests/test_regressions.py", "modified"),
                      ("CHANGELOG", "modified")], ["test_issue11"]),
        rec(3, [12], [("pkg/b.py", "modified"), ("tests/test_split.py", "modified"),
                      ("CHANGELOG", "modified")], ["test_something"]),
        rec(4, [13], [("pkg/a.py", "modified"), ("tests/test_new.py", "added")], ["test_x"]),
    ]


def test_analyze_counts_where_tests_and_changelog_go():
    s = analyze(sample_records())
    assert s["n"] == 4
    t = s["tests"]
    assert (t["touched"], t["added_file"], t["modified_existing"]) == (4, 1, 3)
    assert t["top_files"][0] == ("tests/test_regressions.py", 2)
    assert (t["with_new_tests"], t["issue_named"]) == (4, 2)
    assert s["changelog"]["kind"] == "file" and s["changelog"]["path"] == "CHANGELOG"


def test_fragment_projects_are_recognised_and_get_a_fragment_path():
    recs = [rec(n, [n], [("pkg/a.py", "modified"), (f"changelog/{n}.bugfix.rst", "added")])
            for n in (1, 2, 3)]
    cl = analyze(recs)["changelog"]
    assert cl["kind"] == "fragment" and cl["fragment_dir"] == "changelog"
    assert fragment_path(cl, 843) == "changelog/843.bugfix.rst"
    assert fragment_path({**cl, "fragment_dir": "../x"}, 1) is None
    assert fragment_path({"kind": "file"}, 1) is None


def test_homes_prefer_files_touched_with_the_same_code():
    homes = homes_for_sources(sample_records(), ["pkg/a.py"])
    assert homes == [("tests/test_regressions.py", 2), ("tests/test_split.py", 1)]


def test_changelog_style_and_draft_entry():
    head = (
        "Version 3.2.0\n-------------\n\nUnreleased\n\n"
        "-   Fix the thing. :issue:`10`\n-   Another one. :issue:`11`\n"
    )
    style = changelog_style(head)
    assert style["prefix"] == "-   " and style["ref"] == ":issue:`{n}`"
    entry = draft_changelog_entry(style, 843)
    assert entry.startswith("-   TODO(reviewer):") and entry.endswith(":issue:`843`")
    assert draft_changelog_entry(None, 7).endswith("#7")


def test_convention_findings_name_the_files_past_fixes_used(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("def test_ok(): ...\n")
    (tmp_path / "tests" / "test_issue_5.py").write_text("def test_x(): ...\n")
    recs = [rec(n, [n], [("app.py", "modified"), ("tests/test_app.py", "modified")],
                [f"test_issue{n}_a" if n < 3 else "test_c"]) for n in (1, 2, 3)]
    conv = {"prs": recs, "stats": analyze(recs)}
    found = {f.code: f.message for f in convention_findings(
        conv, tmp_path, ["tests/test_issue_5.py"], ["test_x"], ["app.py"], 5)}
    assert "tests/test_app.py (3 past fixes)" in found["new_test_file"]
    assert "test_issue5_" in found["test_name_convention"]
    assert convention_findings(None, tmp_path, ["t.py"], [], [], 1) == []


def test_checklist_ticks_only_what_was_verified():
    plain = render_checklist(TEMPLATE, {})
    assert plain.startswith("Checklist:") and "<!--" not in plain and "[x]" not in plain
    done = render_checklist(TEMPLATE, ALL_TRUE)
    assert "- [x] Add tests that demonstrate" in done
    assert "- [x] Add an entry in `CHANGES.rst`" in done
    assert "- [x] Run `pre-commit` hooks" in done
    assert "- [ ] Add or update relevant docs" in done
    assert "- [ ] Add `.. versionchanged::`" in done
    assert "- [ ] Run `pytest` and `tox`" in done          # tox was not run
    assert not checklist_tick("Run pytest", {"suite_passed": None})


def test_update_checkboxes_only_adds_ticks():
    text = "Title\n\n- [ ] Add tests that fail without the change\n- [x] My own tick\n- [ ] Docs\n"
    out = update_checkboxes(text, {"tests_added": True, "test_fails_without_fix": True})
    assert "- [x] Add tests that fail" in out and "- [x] My own tick" in out
    assert "- [ ] Docs" in out
    assert update_checkboxes(out, {}) == out


def test_mine_reads_fix_prs_and_learns_the_conventions():
    prs = [
        {"number": 10, "title": "Fix a", "body": "fixes #1", "merged_at": "2026-01-01T00:00:00Z",
         "user": {"login": "dev", "type": "User"}},
        {"number": 11, "title": "bump", "body": "fixes #2", "merged_at": "2026-01-02T00:00:00Z",
         "user": {"login": "dependabot[bot]", "type": "Bot"}},
        {"number": 12, "title": "unmerged", "body": "fixes #3", "merged_at": None,
         "user": {"login": "dev", "type": "User"}},
        {"number": 13, "title": "a feature", "body": "nothing", "merged_at": "2026-01-03T00:00:00Z",
         "user": {"login": "dev", "type": "User"}},
    ]
    files = [
        {"filename": "pkg/a.py", "status": "modified", "additions": 2, "deletions": 1},
        {"filename": "tests/test_regressions.py", "status": "modified", "additions": 8,
         "deletions": 0, "patch": "@@ -1 +1,9 @@\n+def test_issue1_a():\n+    assert True\n"},
        {"filename": "CHANGELOG", "status": "modified", "additions": 1, "deletions": 0},
    ]
    routes = {
        "/repos/o/r": {"default_branch": "main", "archived": False, "disabled": False,
                       "private": False},
        "/repos/o/r/pulls": prs,
        "/repos/o/r/pulls/10/files": files,
    }
    texts = {"/repos/o/r/contents/.github/pull_request_template.md": "Checklist:\n\n- [ ] Tests\n"}
    conv = mine(mock_client(routes, texts), "o/r", now=NOW)
    assert [p["number"] for p in conv["prs"]] == [10]
    r = conv["prs"][0]
    assert r["issues"] == [1] and r["new_tests"] == ["test_issue1_a"]
    assert [f["kind"] for f in r["files"]] == ["source", "test", "changelog"]
    assert conv["pr_template"].startswith("Checklist:")
    assert conv["stats"]["tests"]["issue_named"] == 1


def test_load_or_mine_caches_for_a_week(tmp_path):
    routes = {
        "/repos/o/r": {"default_branch": "main", "archived": False, "disabled": False,
                       "private": False},
        "/repos/o/r/pulls": [],
    }
    calls: list[str] = []
    first = load_or_mine(mock_client(routes, counter=calls), "o/r", tmp_path, now=NOW)
    assert first["prs"] == [] and calls
    boom = github_client("", transport=httpx.MockTransport(lambda r: pytest.fail("network used")))
    assert load_or_mine(boom, "o/r", tmp_path, now=NOW)["repo"] == "o/r"
    again: list[str] = []
    load_or_mine(mock_client(routes, counter=again), "o/r", tmp_path, refresh=True, now=NOW)
    assert again
