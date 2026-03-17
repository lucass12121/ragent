ERROR_PATTERNS = {
    "IndexError": (
        "Check that your index is within the valid range. "
        "Use len() to verify the sequence length before accessing elements. "
        "Consider using try/except to handle out-of-range access."
    ),
    "TypeError": (
        "Verify the types of operands and function arguments. "
        "Use type() or isinstance() to inspect values. "
        "Ensure you are not mixing incompatible types in operations."
    ),
    "ValueError": (
        "Check that the value is appropriate for the operation. "
        "Validate input before passing it to functions like int() or float(). "
        "Consider using try/except around the conversion."
    ),
    "NameError": (
        "Make sure the variable or function is defined before use. "
        "Check for typos in variable names. "
        "Verify that imports are correct and at the top of the file."
    ),
    "AttributeError": (
        "Verify the object has the attribute you are accessing. "
        "Use hasattr() or dir() to inspect available attributes. "
        "Check that the object is the expected type and not None."
    ),
    "KeyError": (
        "Check that the key exists in the dictionary before accessing it. "
        "Use dict.get(key, default) for safe access. "
        "Print the dictionary keys to verify available entries."
    ),
    "ZeroDivisionError": (
        "Add a check to ensure the divisor is not zero before dividing. "
        "Use a conditional or try/except to handle the zero case. "
        "Consider what the correct fallback value should be."
    ),
}


def search_error(error_msg: str) -> str:
    """Match an error message to known patterns and return fix suggestions."""
    for error_type, suggestion in ERROR_PATTERNS.items():
        if error_type in error_msg:
            return suggestion

    return (
        "Review the error message and traceback carefully. "
        "Search the Python documentation or Stack Overflow for the specific error. "
        "Add print statements to inspect variable values around the failing line."
    )
