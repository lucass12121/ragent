import subprocess
import sys
import tempfile
import os


def execute_code(code: str, timeout: int = 5) -> dict:
    """Execute Python code in a subprocess and return results."""
    result = {"stdout": "", "stderr": "", "success": False, "timeout": False}

    # Write code to a temp file
    fd, tmp_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)

        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        result["success"] = proc.returncode == 0 and len(proc.stdout.strip()) > 0
    except subprocess.TimeoutExpired:
        result["timeout"] = True
    finally:
        os.unlink(tmp_path)

    return result
