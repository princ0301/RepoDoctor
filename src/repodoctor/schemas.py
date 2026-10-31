from pydantic import BaseModel, Field

class Edit(BaseModel):
    """One exact search/replace edit to a file in the repo."""

    path: str = Field(description="File path relative to the repo root, forward slashes")
    old: str = Field(
        description="Exact existing text to replace. Must match the file exactly once. "
        "Use an empty string only to create a new file."
    )
    new: str = Field(description="Replacement text")