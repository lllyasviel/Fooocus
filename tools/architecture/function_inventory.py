#!/usr/bin/env python3
"""Generate a Python function/class inventory for architecture refactors.

Usage:
    python tools/architecture/function_inventory.py \
        --root . \
        --out docs/experimental-redesign/generated-function-inventory.md \
        --json docs/experimental-redesign/generated-function-inventory.json

The script uses Python's built-in AST parser. It does not import project modules,
so it can run without GPU dependencies or model files.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".github",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "models",
    "outputs",
    "temp",
    "cache",
}


@dataclass
class FunctionInfo:
    name: str
    qualname: str
    kind: str
    line: int
    end_line: int | None
    args: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    docstring: bool = False


@dataclass
class ClassInfo:
    name: str
    qualname: str
    line: int
    end_line: int | None
    bases: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    docstring: bool = False
    methods: list[FunctionInfo] = field(default_factory=list)


@dataclass
class ModuleInfo:
    path: str
    import_count: int = 0
    docstring: bool = False
    parse_error: str | None = None
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return dotted_name(node.func)
    if isinstance(node, ast.Subscript):
        return dotted_name(node.value)
    if isinstance(node, ast.Constant):
        return repr(node.value)
    return type(node).__name__


def function_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args: list[str] = []
    for arg in node.args.posonlyargs:
        args.append(arg.arg)
    for arg in node.args.args:
        args.append(arg.arg)
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    for arg in node.args.kwonlyargs:
        args.append(arg.arg)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    return args


def decorators(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    return [dotted_name(item) for item in node.decorator_list]


def build_function_info(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    module_path: str,
    parent: str | None = None,
) -> FunctionInfo:
    kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
    if parent:
        kind = "async_method" if isinstance(node, ast.AsyncFunctionDef) else "method"
    qualname = f"{module_path}:{parent + '.' if parent else ''}{node.name}"
    return FunctionInfo(
        name=node.name,
        qualname=qualname,
        kind=kind,
        line=node.lineno,
        end_line=getattr(node, "end_lineno", None),
        args=function_args(node),
        decorators=decorators(node),
        docstring=bool(ast.get_docstring(node)),
    )


def direct_child_functions(nodes: Iterable[ast.stmt]) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [node for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def direct_child_classes(nodes: Iterable[ast.stmt]) -> list[ast.ClassDef]:
    return [node for node in nodes if isinstance(node, ast.ClassDef)]


def parse_module(path: Path, root: Path) -> ModuleInfo:
    rel = path.relative_to(root).as_posix()
    info = ModuleInfo(path=rel)
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except Exception as exc:  # noqa: BLE001 - inventory should record parse failures
        info.parse_error = f"{type(exc).__name__}: {exc}"
        return info

    info.docstring = bool(ast.get_docstring(tree))
    info.import_count = sum(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body)

    for node in direct_child_functions(tree.body):
        info.functions.append(build_function_info(node, rel))

    for class_node in direct_child_classes(tree.body):
        class_info = ClassInfo(
            name=class_node.name,
            qualname=f"{rel}:{class_node.name}",
            line=class_node.lineno,
            end_line=getattr(class_node, "end_lineno", None),
            bases=[dotted_name(base) for base in class_node.bases],
            decorators=decorators(class_node),
            docstring=bool(ast.get_docstring(class_node)),
        )
        for method_node in direct_child_functions(class_node.body):
            class_info.methods.append(build_function_info(method_node, rel, parent=class_node.name))
        info.classes.append(class_info)

    return info


def iter_python_files(root: Path, exclude_dirs: set[str]) -> Iterable[Path]:
    for path in sorted(root.rglob("*.py")):
        parts = set(path.relative_to(root).parts)
        if parts & exclude_dirs:
            continue
        yield path


def render_markdown(modules: list[ModuleInfo]) -> str:
    total_functions = sum(len(module.functions) for module in modules)
    total_classes = sum(len(module.classes) for module in modules)
    total_methods = sum(len(cls.methods) for module in modules for cls in module.classes)
    parse_errors = [module for module in modules if module.parse_error]

    lines: list[str] = []
    lines.append("# Generated Function Inventory")
    lines.append("")
    lines.append("Generated by `tools/architecture/function_inventory.py`.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|---|---:|")
    lines.append(f"| Python modules scanned | {len(modules)} |")
    lines.append(f"| Top-level functions | {total_functions} |")
    lines.append(f"| Classes | {total_classes} |")
    lines.append(f"| Methods | {total_methods} |")
    lines.append(f"| Parse errors | {len(parse_errors)} |")
    lines.append("")

    if parse_errors:
        lines.append("## Parse Errors")
        lines.append("")
        lines.append("| Module | Error |")
        lines.append("|---|---|")
        for module in parse_errors:
            lines.append(f"| `{module.path}` | `{module.parse_error}` |")
        lines.append("")

    lines.append("## Modules")
    lines.append("")

    for module in modules:
        lines.append(f"### `{module.path}`")
        lines.append("")
        if module.parse_error:
            lines.append(f"Parse error: `{module.parse_error}`")
            lines.append("")
            continue

        lines.append("| Module Info | Value |")
        lines.append("|---|---:|")
        lines.append(f"| Imports | {module.import_count} |")
        lines.append(f"| Has module docstring | {str(module.docstring)} |")
        lines.append(f"| Top-level functions | {len(module.functions)} |")
        lines.append(f"| Classes | {len(module.classes)} |")
        lines.append("")

        if module.functions:
            lines.append("#### Top-Level Functions")
            lines.append("")
            lines.append("| Function | Line | Args | Decorators | Docstring |")
            lines.append("|---|---:|---|---|---:|")
            for fn in module.functions:
                args = ", ".join(fn.args)
                decs = ", ".join(fn.decorators)
                lines.append(f"| `{fn.name}` | {fn.line} | `{args}` | `{decs}` | {fn.docstring} |")
            lines.append("")

        if module.classes:
            lines.append("#### Classes")
            lines.append("")
            for cls in module.classes:
                bases = ", ".join(cls.bases)
                decs = ", ".join(cls.decorators)
                lines.append(f"##### `{cls.name}`")
                lines.append("")
                lines.append("| Class Info | Value |")
                lines.append("|---|---|")
                lines.append(f"| Line | {cls.line} |")
                lines.append(f"| Bases | `{bases}` |")
                lines.append(f"| Decorators | `{decs}` |")
                lines.append(f"| Has docstring | {cls.docstring} |")
                lines.append("")
                if cls.methods:
                    lines.append("| Method | Line | Args | Decorators | Docstring |")
                    lines.append("|---|---:|---|---|---:|")
                    for method in cls.methods:
                        args = ", ".join(method.args)
                        mdecs = ", ".join(method.decorators)
                        lines.append(f"| `{method.name}` | {method.line} | `{args}` | `{mdecs}` | {method.docstring} |")
                    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate AST-based function inventory.")
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    parser.add_argument("--out", default="docs/experimental-redesign/generated-function-inventory.md", help="Markdown output path.")
    parser.add_argument("--json", default="docs/experimental-redesign/generated-function-inventory.json", help="JSON output path.")
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Additional directory name to exclude. Can be provided multiple times.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    exclude_dirs = DEFAULT_EXCLUDE_DIRS | set(args.exclude_dir)

    modules = [parse_module(path, root) for path in iter_python_files(root, exclude_dirs)]

    out_path = Path(args.out)
    json_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(render_markdown(modules), encoding="utf-8")
    json_path.write_text(json.dumps([asdict(module) for module in modules], indent=2), encoding="utf-8")

    print(f"Scanned {len(modules)} Python modules")
    print(f"Wrote {out_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
