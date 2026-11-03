"""Find the source of a maintainers' test from its JUnit-style id."""

import ast


def function_source(source: str, classes: list[str], name: str) -> str | None:
    """Source (decorators included) of a test function or method."""
    scope = ast.parse(source).body
    for cls in classes:
        node = next((n for n in scope if isinstance(n, ast.ClassDef) and n.name == cls), None)
        if node is None:
            return None
        scope = node.body
    for n in scope:
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == name:
            start = min([n.lineno, *(d.lineno for d in n.decorator_list)])
            return "\n".join(source.splitlines()[start - 1 : n.end_lineno])
    return None


def gold_test_source(test_id: str, texts: dict[str, str]) -> tuple[str, str] | None:
    """(file, source) for ids like 'tests.test_a.TestA::test_m[1]', or None if not found."""
    classname, _, name = test_id.partition("::")
    name = name.split("[", 1)[0]
    parts = classname.split(".")
    for k in range(len(parts), 0, -1):
        path = "/".join(parts[:k]) + ".py"
        if path in texts:
            src = function_source(texts[path], parts[k:], name)
            return (path, src) if src else None
    return None
