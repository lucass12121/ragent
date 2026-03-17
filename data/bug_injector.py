"""Inject bugs into correct Python code for training data generation."""

import random
import re


class BugInjector:
    """Injects one of three bug types into Python source code."""

    BUG_TYPES = ["index_error", "type_error", "logic_error"]

    def inject(self, code: str, bug_type: str = "random") -> str:
        """Inject a bug into the given code.

        Args:
            code: Source code string.
            bug_type: One of "index_error", "type_error", "logic_error", or "random".

        Returns:
            Modified code with an injected bug.
        """
        if bug_type == "random":
            bug_type = random.choice(self.BUG_TYPES)

        injector = {
            "index_error": self._inject_index_error,
            "type_error": self._inject_type_error,
            "logic_error": self._inject_logic_error,
        }[bug_type]

        result = injector(code)
        # If the chosen injector couldn't find a pattern, fall back to logic_error
        # which can always apply (operator replacement is very common).
        if result == code and bug_type != "logic_error":
            result = self._inject_logic_error(code)
        if result == code:
            result = self._inject_fallback(code)
        return result

    # ------------------------------------------------------------------
    # Index error: arr[i] -> arr[i+1]  (off-by-one → potential IndexError)
    # ------------------------------------------------------------------
    def _inject_index_error(self, code: str) -> str:
        # Match patterns like var[expr] but not function definitions / slices
        pattern = re.compile(r'(\w+)\[([^\[\]:]+)\]')
        matches = list(pattern.finditer(code))
        if not matches:
            return code
        m = random.choice(matches)
        idx_expr = m.group(2).strip()
        new_expr = f"{m.group(1)}[{idx_expr} + 1]"
        return code[:m.start()] + new_expr + code[m.end():]

    # ------------------------------------------------------------------
    # Type error: inject a string + int concatenation
    # ------------------------------------------------------------------
    def _inject_type_error(self, code: str) -> str:
        # Find `return <expr>` and wrap it so a string is concatenated with the value
        pattern = re.compile(r'(return\s+)(.+)')
        matches = list(pattern.finditer(code))
        if not matches:
            return code
        m = random.choice(matches)
        original_expr = m.group(2).strip()
        new_expr = f'"result: " + {original_expr}'
        return code[:m.start()] + f"return {new_expr}" + code[m.end():]

    # ------------------------------------------------------------------
    # Logic error: flip a comparison or arithmetic operator
    # ------------------------------------------------------------------
    _OP_SWAPS = {
        ">=": "<",
        "<=": ">",
        "==": "!=",
        "!=": "==",
        " > ": " >= ",
        " < ": " <= ",
        " + ": " - ",
        " - ": " + ",
        " * ": " / ",
    }

    def _inject_logic_error(self, code: str) -> str:
        # Try operators in random order
        ops = list(self._OP_SWAPS.items())
        random.shuffle(ops)
        for old_op, new_op in ops:
            if old_op in code:
                # Replace only the first occurrence
                return code.replace(old_op, new_op, 1)
        return code

    # ------------------------------------------------------------------
    # Fallback: guarantee the code is changed even if no pattern matched
    # ------------------------------------------------------------------
    @staticmethod
    def _inject_fallback(code: str) -> str:
        # Insert an off-by-one in the first `range(` call, or append a bad line
        if "range(" in code:
            return code.replace("range(", "range(1 + ", 1)
        # Last resort: add a line that will cause a NameError
        return code + "\n    _undefined_var\n"
