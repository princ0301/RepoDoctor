import re

from pydantic import BaseModel, Field, field_validator


class Edit(BaseModel):
    """One exact search/replace edit to a file in the repo."""

    path: str = Field(description="File path relative to the repo root, forward slashes")
    old: str = Field(
        description="Exact existing text to replace. Must match the file exactly once. "
        "Use an empty string only to create a new file."
    )
    new: str = Field(description="Replacement text")

class ExploreResult(BaseModel):
    relevant_files: list[str] = Field(
        description="Up to 5 source files (paths relative to the repo root) that most "
        "likely need to change, most important first. Only files you have actually seen."
    )
    hypothesis: str = Field(description="2-4 sentences on the root cause of the bug")


class FixProposal(BaseModel):
    reasoning: str = Field(default="", description="Brief explanation of the fix")
    edits: list[Edit] = Field(default_factory=list, description="Exact search/replace edits")
    need_more_files: list[str] = Field(
        default_factory=list,
        description="Only if you cannot write edits without seeing other files: their paths "
        "(relative to the repo root). Leave edits empty in that case.",
    )

    @field_validator("edits", "need_more_files", mode="before")
    @classmethod
    def _lenient_lists(cls, v, info):
        """Accept null for either list, and 'a.py, b.py' for need_more_files."""
        if v is None:
            return []
        if info.field_name == "need_more_files" and isinstance(v, str):
            return [p.strip() for p in re.split(r"[,\n]", v) if p.strip()]
        return v

class ReproTest(BaseModel):
    reasoning: str = Field(description="What the test checks and why the bug makes it fail")
    test_name: str = Field(description="Name of the test function; must start with 'test'")
    code: str = Field(description="Complete contents of a NEW pytest test file")


    
