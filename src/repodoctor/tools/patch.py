from dataclasses import dataclass, field
from pathlib import Path
from repodoctor.schemas import Edit

@dataclass
class PatchResult:
    ok: bool
    changed_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # written for an LLM to read

@dataclass
class _File:
    text: str  # always LF-normalized in memory
    crlf: bool = False
    is_new: bool = False

def _norm(s: str) -> str:
    return s.replace("\r\n", "\n")

def _resolve(root: Path, rel: str) -> Path | None:
    """Resolve rel under root; None if it escape root or touches .git."""
    p = (root / rel).resolve()
    if not p.is_relative_to(root):
        return None
    if ".git" in p.relative_to(root).parts:
        return None
    return p

def _load(path: Path) -> _File:
    raw = path.read_bytes().decode("utf-8")
    return _File(text=_norm(raw), crlf="\r\n" in raw)

def apply_edits(root: Path, edits: list[Edit]) -> PatchResult:
    """Apply search/replace edits automatically. Nothing is written if any edit fails."""
    root = root.resolve()
    files: dict[Path, _File] = {}
    errors: list[str] = []

    for i, e in enumerate(edits, 1):
        label = f"edit {i} ({e.path})"
        path = _resolve(root, e.path)
        if path is None:
            errors.append(f"{label}: path is outside the repository or inside .git")
            continue

        if e.old == "":  # create a new file
            if path in files or path.exists():
                errors.append(
                    f"{label}: file already exists. To change it, set 'old' to the "
                    "exact existing text you want to replace."
                )
            else:
                files[path] = _File(text=_norm(e.new), is_new=True)
            continue

        f = files.get(path)
        if f is None:
            if not path.is_file():
                errors.append(f"{label}: file does not exist")
                continue
            try:
                f = files[path] = _load(path)
            except UnicodeDecodeError:
                errors.append(f"{label}: not a UTF-8 text file")
                continue

        old, new = _norm(e.old), _norm(e.new)
        count = f.text.count(old)

        if count == 0:
            errors.append(
                f"{label}: 'old' text not found. It must match the file exactly, "
                "including whitespace and indentation. Re-read the file and copy it exactly."
            )
        elif count > 1:
            errors.append(
                f"{label}: 'old' text appears {count} times. Include more surrounding "
                "lines so it matches exactly once."
            )
        else:
            f.text = f.text.replace(old, new, 1)

    if errors:
        return PatchResult(ok=False, errors=errors)

    for path, f in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        text = f.text.replace("\n", "\r\n") if f.crlf else f.text
        path.write_bytes(text.encode("utf-8"))

    changed = sorted(p.relative_to(root).as_posix() for p in files)
    return PatchResult(ok=True, changed_files=changed)