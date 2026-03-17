from tools.executor import execute_code
from tools.searcher import search_error
from tools.patcher import patch_code

TOOL_REGISTRY = {
    "execute_code": execute_code,
    "search_error": search_error,
    "patch_code": patch_code,
}
