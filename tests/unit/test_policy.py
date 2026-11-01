from repodoctor.tools.policy import policy_mentions


def test_finds_ai_mentions_with_file_and_line(tmp_path):
    (tmp_path / "CONTRIBUTING.md").write_text(
        "Be nice.\nAI-generated contributions are not accepted.\n"
    )
    assert policy_mentions(tmp_path) == [
        "CONTRIBUTING.md:2: AI-generated contributions are not accepted."
    ]


def test_finds_mentions_in_the_github_directory(tmp_path):
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "CONTRIBUTING.md").write_text("Using LLMs is discouraged.\n")
    assert policy_mentions(tmp_path)[0].startswith(".github/CONTRIBUTING.md:1:")


def test_no_mentions_and_missing_files(tmp_path):
    assert policy_mentions(tmp_path) == []
    (tmp_path / "CONTRIBUTING.md").write_text("Run pytest before you push.\n")
    assert policy_mentions(tmp_path) == []


def test_ordinary_words_are_not_matches(tmp_path):
    (tmp_path / "CONTRIBUTING.md").write_text("We maintain main, said Bob. Fair play.\n")
    assert policy_mentions(tmp_path) == []


def test_number_of_hits_is_capped(tmp_path):
    (tmp_path / "CONTRIBUTING.md").write_text("AI is mentioned here.\n" * 50)
    assert len(policy_mentions(tmp_path, max_hits=3)) == 3