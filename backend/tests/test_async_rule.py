"""The asynchronous rule, enforced mechanically.

The specification is explicit: pandas and DuckDB operations are synchronous and
blocking, and must be wrapped in `asyncio.to_thread` so they cannot stall the event
loop. That is easy to state and easy to violate six months later, so it is checked
here rather than trusted.

The check is deliberately narrow: **inside an `async def`, no call may reach
pandas / numpy / duckdb / yfinance directly.** The sanctioned pattern is to define a
plain `def work()` inside the coroutine and hand it to `run_blocking`, so sync helpers
(including nested ones) are exempt. That is what makes the rule testable at all — a
blanket ban on importing pandas would flag the helpers that do the actual work.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app"

#: Modules whose functions block. Reaching any of these from a coroutine body without
#: going through `run_blocking` stalls every other request on the server.
BLOCKING_MODULES = {"pandas", "numpy", "duckdb", "yfinance", "scipy", "statsmodels"}


def _module_aliases(tree: ast.Module) -> dict[str, str]:
    """Map local name -> real module name for top-level imports."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                aliases[alias.asname or root] = root
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            aliases[node.module.split(".")[0]] = node.module.split(".")[0]
    return aliases


def _calls_reaching_blocking_libs(
    func: ast.AsyncFunctionDef, aliases: dict[str, str]
) -> list[tuple[int, str]]:
    """Calls inside `func` that touch a blocking library, excluding nested sync defs."""
    found: list[tuple[int, str]] = []

    def visit(node: ast.AST, inside_nested_sync: bool) -> None:
        for child in ast.iter_child_nodes(node):
            # A nested plain `def` is the sanctioned `def work()` helper. Its body runs
            # on a worker thread, so descending into it would defeat the whole test.
            if isinstance(child, ast.FunctionDef) and not inside_nested_sync:
                continue
            if isinstance(child, ast.Call):
                target = child.func
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                    module = aliases.get(target.value.id)
                    if module in BLOCKING_MODULES:
                        found.append((child.lineno, f"{target.value.id}.{target.attr}"))
            visit(child, inside_nested_sync)

    visit(func, False)
    return found


def _async_functions(path: Path) -> list[tuple[ast.AsyncFunctionDef, dict[str, str]]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases = _module_aliases(tree)
    return [(node, aliases) for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)]


def _app_files() -> list[Path]:
    return sorted(p for p in APP.rglob("*.py") if p.name != "blocking.py")


def test_no_coroutine_calls_a_blocking_library_directly() -> None:
    offenders: list[str] = []
    for path in _app_files():
        for func, aliases in _async_functions(path):
            for lineno, expression in _calls_reaching_blocking_libs(func, aliases):
                offenders.append(
                    f"{path.relative_to(APP.parent)}:{lineno} in {func.name}() -> {expression}"
                )

    assert offenders == [], (
        "These coroutines call a blocking library without run_blocking:\n  "
        + "\n  ".join(offenders)
    )


def test_the_scan_would_actually_catch_a_violation(tmp_path: Path) -> None:
    """A guard test that cannot fail is worse than none.

    This feeds the checker a file that *does* violate the rule and asserts it is
    reported, so a future refactor that silently breaks the detector is caught.
    """
    sample = tmp_path / "bad.py"
    sample.write_text(
        "import pandas as pd\n"
        "\n"
        "\n"
        "async def handler():\n"
        "    return pd.read_parquet('x.parquet')\n",
        encoding="utf-8",
    )
    functions = _async_functions(sample)
    assert len(functions) == 1
    func, aliases = functions[0]
    assert _calls_reaching_blocking_libs(func, aliases)


def test_a_sanctioned_work_helper_is_not_flagged(tmp_path: Path) -> None:
    sample = tmp_path / "good.py"
    sample.write_text(
        "import pandas as pd\n"
        "from app.core.blocking import run_blocking\n"
        "\n"
        "\n"
        "async def handler():\n"
        "    def work():\n"
        "        return pd.read_parquet('x.parquet')\n"
        "\n"
        "    return await run_blocking(work)\n",
        encoding="utf-8",
    )
    func, aliases = _async_functions(sample)[0]
    assert _calls_reaching_blocking_libs(func, aliases) == []


def test_run_blocking_uses_to_thread() -> None:
    """The chokepoint must genuinely leave the event loop.

    `run_blocking` calling the function inline would satisfy every other test in this
    file while blocking exactly as before.
    """
    source = (APP / "core" / "blocking.py").read_text(encoding="utf-8")
    assert "asyncio.to_thread" in source

    tree = ast.parse(source, filename="blocking.py")
    coroutines = [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)]
    assert coroutines, "run_blocking must itself be a coroutine"
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "to_thread"
        for coroutine in coroutines
        for node in ast.walk(coroutine)
    )


@pytest.mark.parametrize("path", _app_files(), ids=lambda p: p.name)
def test_every_service_entry_point_is_async(path: Path) -> None:
    """A public method on a service that is not async would be called from a route
    without an await, blocking the loop with no wrapper in between."""
    if path.parent.name != "services":
        return
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.endswith("Service"):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                    raise AssertionError(
                        f"{path.name}: {node.name}.{item.name} is not async. "
                        "Service entry points must be coroutines that route their "
                        "blocking work through run_blocking."
                    )
