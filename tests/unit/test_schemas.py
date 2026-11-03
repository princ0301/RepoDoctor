import pytest
from pydantic import ValidationError

from repodoctor.schemas import FixProposal, ExploreResult


def test_fix_proposal_tolerates_sloppy_model_output():
    p = FixProposal.model_validate(
        {"reasoning": "r", "edits": None, "need_more_files": "a.py, b/c.py"}
    )
    assert p.edits == [] and p.need_more_files == ["a.py", "b/c.py"]


def test_fix_proposal_still_rejects_wrong_structures():
    with pytest.raises(ValidationError):
        FixProposal.model_validate({"edits": 5})

def test_explore_result_pattern_defaults_to_empty():
    assert ExploreResult(relevant_files=[], hypothesis="h", pattern=None).pattern == ""
    assert ExploreResult(relevant_files=[], hypothesis="h").pattern == ""

