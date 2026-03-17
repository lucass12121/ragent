import ast


def patch_code(original: str, patched: str) -> str:
    """Return patched code if it is valid Python, otherwise return original."""
    try:
        ast.parse(patched)
        return patched
    except SyntaxError:
        print(f"Warning: patched code has syntax errors, returning original code.")
        return original
