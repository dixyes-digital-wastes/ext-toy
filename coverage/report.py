#!/usr/bin/env python3
"""Aggregate gcov JSON coverage for the toy extension sources.

The build must use --coverage. Run from anywhere; paths resolve against the
repository root, which is the parent directory of this script.

gcov reports one JSON file per object. Only .c files in the repository root and
under codegen/ are counted. Header files, tests and generated machine code are
not instrumented by this build.

Windows-only preprocessor regions are tagged so their lines and branches do not
count as Linux coverage gaps. The condition evaluator knows the macros this
project uses; an unknown condition is treated as active on both platforms,
which under-reports Windows-only regions but never hides a real gap.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MACROS = {
    "_WIN32": (False, True),
    "_MSC_VER": (False, True),
    "_M_X64": (False, True),
    "_M_IX86": (False, False),
    "__GNUC__": (True, False),
    "__clang__": (True, False),
    "__x86_64__": (True, False),
    "__i386__": (False, False),
    "__cplusplus": (False, False),
    "TOY_WIN64_ABI": (False, True),
    "TOY_ISA_MSVC_X64": (False, True),
    "TOY_ISA_ARCH_X86": (True, True),
    "ZTS": (False, False),
    "COMPILE_DL_TOY": (True, True),
    "ZEND_COMPILE_DL_EXT": (False, False),
    "HAVE_CONFIG_H": (True, False),
    "MAP_NORESERVE": (True, False),
}


class UnknownCondition(Exception):
    pass


def _eval_profile(condition: str, profile: int) -> bool:
    expression = condition
    expression = re.sub(r"defined\s*\(\s*([A-Za-z_]\w*)\s*\)", r"DEF('\1')", expression)
    expression = re.sub(r"defined\s+([A-Za-z_]\w*)", r"DEF('\1')", expression)
    expression = expression.replace("&&", " and ").replace("||", " or ")

    def define(name: str) -> bool:
        if name not in MACROS:
            raise UnknownCondition(name)
        return MACROS[name][profile]

    namespace = {
        "DEF": define,
        "and": None,
        "or": None,
        "not": None,
    }
    try:
        return bool(eval(expression, {"__builtins__": {}}, namespace))
    except (NameError, SyntaxError, TypeError):
        raise UnknownCondition(condition)


def _branch_value(condition: str) -> tuple[bool | None, bool | None]:
    try:
        return _eval_profile(condition, 0), _eval_profile(condition, 1)
    except UnknownCondition:
        return None, None


def _strip_comment(text: str) -> str:
    cut = text.find("//")
    if cut >= 0:
        text = text[:cut]
    start = text.find("/*")
    if start >= 0:
        end = text.find("*/", start + 2)
        text = text[:start] + (text[end + 2:] if end >= 0 else "")
    return text.strip()


class Frame:
    def __init__(self, parent_l: bool, parent_w: bool, branch_l: bool | None,
                 branch_w: bool | None) -> None:
        known_l = True if branch_l is None else branch_l
        known_w = True if branch_w is None else branch_w
        self.parent_l = parent_l
        self.parent_w = parent_w
        self.active_l = parent_l and known_l
        self.active_w = parent_w and known_w
        self.taken_l = known_l
        self.taken_w = known_w


def windows_only_lines(path: Path) -> set[int]:
    result: set[int] = set()
    stack: list[Frame] = []
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()

    for number, raw in enumerate(text, 1):
        stripped = raw.strip()
        if stripped.startswith("#"):
            match = re.match(r"#\s*(\w+)(.*)", stripped)
            if match:
                directive = match.group(1)
                rest = _strip_comment(match.group(2).strip())
                if directive in ("if", "ifdef", "ifndef"):
                    if directive == "ifdef":
                        rest = f"defined({rest})"
                    elif directive == "ifndef":
                        rest = f"!defined({rest})"
                    branch_l, branch_w = _branch_value(rest)
                    parent_l = stack[-1].active_l if stack else True
                    parent_w = stack[-1].active_w if stack else True
                    stack.append(Frame(parent_l, parent_w, branch_l, branch_w))
                    continue
                if directive == "elif":
                    if not stack:
                        continue
                    frame = stack[-1]
                    branch_l, branch_w = _branch_value(rest)
                    known_l = True if branch_l is None else branch_l
                    known_w = True if branch_w is None else branch_w
                    frame.active_l = frame.parent_l and not frame.taken_l and known_l
                    frame.active_w = frame.parent_w and not frame.taken_w and known_w
                    frame.taken_l = frame.taken_l or known_l
                    frame.taken_w = frame.taken_w or known_w
                    continue
                if directive == "else":
                    if not stack:
                        continue
                    frame = stack[-1]
                    frame.active_l = frame.parent_l and not frame.taken_l
                    frame.active_w = frame.parent_w and not frame.taken_w
                    frame.taken_l = frame.taken_w = True
                    continue
                if directive == "endif":
                    if stack:
                        stack.pop()
                    continue

        if any(frame.active_w and not frame.active_l for frame in stack) and stripped:
            result.add(number)

    return result


def relative_source(source: Path) -> Path | None:
    try:
        relative = source.resolve().relative_to(ROOT)
    except ValueError:
        return None
    if relative.suffix != ".c":
        return None
    if relative.parts and relative.parts[0] == "tests":
        return None
    return relative


def collect() -> dict[str, dict]:
    gcda_files = sorted(path for path in ROOT.glob("**/*.gcda") if "a-conftest" not in path.name)
    if not gcda_files:
        raise SystemExit("no .gcda files found; build with --coverage and run the tests first")

    files: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="toy-gcov-") as tmp:
        tmp_path = Path(tmp)
        for gcda in gcda_files:
            output = tmp_path / f"{gcda.stem}.gcov.json.gz"
            if output.exists():
                output.unlink()
            result = subprocess.run(
                ["gcov", "--json-format", "-b", "-c", str(gcda.resolve())],
                cwd=tmp_path, capture_output=True, text=True)
            if result.returncode != 0 or not output.exists():
                print(f"gcov failed for {gcda}: {result.stderr.strip()}", file=sys.stderr)
                continue
            with gzip.open(output, "rt") as handle:
                document = json.load(handle)
            for entry in document["files"]:
                relative = relative_source(Path(entry["file"]))
                if relative is None:
                    continue
                if str(relative) in files:
                    continue
                files[str(relative)] = entry
    return files


def summarize(files: dict[str, dict]) -> dict:
    summary = {"files": {}, "totals": {
        "lines": {"total": 0, "covered": 0},
        "branches": {"total": 0, "covered": 0},
        "functions": {"total": 0, "covered": 0},
    }}

    for name in sorted(files):
        entry = files[name]
        windows = windows_only_lines(ROOT / name)
        lines_total = lines_covered = 0
        branches_total = branches_covered = 0
        uncovered_lines: list[int] = []
        uncovered_windows_lines: list[int] = []
        for line in entry["lines"]:
            number = line["line_number"]
            count = line["count"]
            lines_total += 1
            if count > 0:
                lines_covered += 1
            elif number in windows:
                uncovered_windows_lines.append(number)
            else:
                uncovered_lines.append(number)
            for branch in line.get("branches", []):
                branches_total += 1
                if branch.get("count", 0) > 0:
                    branches_covered += 1

        functions_total = len(entry["functions"])
        functions_covered = sum(1 for function in entry["functions"]
                                if function.get("execution_count", 0) > 0)
        uncovered_functions = [function["name"] for function in entry["functions"]
                               if function.get("execution_count", 0) == 0]

        file_summary = {
            "lines": {"total": lines_total, "covered": lines_covered},
            "branches": {"total": branches_total, "covered": branches_covered},
            "functions": {"total": functions_total, "covered": functions_covered},
            "uncovered_lines": uncovered_lines,
            "uncovered_windows_lines": uncovered_windows_lines,
            "uncovered_functions": uncovered_functions,
            "windows_only_lines": sorted(windows),
            "windows_only_count": len(windows),
        }
        summary["files"][name] = file_summary
        for key in ("lines", "branches", "functions"):
            summary["totals"][key]["total"] += file_summary[key]["total"]
            summary["totals"][key]["covered"] += file_summary[key]["covered"]

    return summary


def percent(covered: int, total: int) -> str:
    if total == 0:
        return "   -  "
    return f"{100.0 * covered / total:5.1f}%"


def compact_ranges(numbers: list[int]) -> str:
    if not numbers:
        return "-"
    numbers = sorted(set(numbers))
    ranges: list[str] = []
    start = previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append(f"{start}" if start == previous else f"{start}-{previous}")
        start = previous = number
    ranges.append(f"{start}" if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def print_table(summary: dict) -> None:
    print(f"{'file':24} {'lines':>16} {'branches':>16} {'functions':>16}")
    for name, entry in summary["files"].items():
        cells = []
        for key in ("lines", "branches", "functions"):
            item = entry[key]
            cells.append(f"{item['covered']:4}/{item['total']:<4} {percent(item['covered'], item['total'])}")
        print(f"{name:24} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16}")
    totals = summary["totals"]
    cells = []
    for key in ("lines", "branches", "functions"):
        item = totals[key]
        cells.append(f"{item['covered']:4}/{item['total']:<4} {percent(item['covered'], item['total'])}")
    print(f"{'TOTAL':24} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16}")


def print_uncovered(summary: dict) -> None:
    print("\nuncovered lines by file (Windows-only lines excluded):")
    for name, entry in summary["files"].items():
        if entry["uncovered_lines"]:
            print(f"  {name}: {compact_ranges(entry['uncovered_lines'])}")
    print("\nuncovered functions:")
    found = False
    for name, entry in summary["files"].items():
        for function in entry["uncovered_functions"]:
            print(f"  {name}: {function}")
            found = True
    if not found:
        print("  (none)")
    print("\nWindows-only source lines (not required on Linux):")
    windows = 0
    for name, entry in summary["files"].items():
        if entry["windows_only_lines"]:
            windows += entry["windows_only_count"]
            print(f"  {name}: {compact_ranges(entry['windows_only_lines'])}")
    print(f"  total: {windows}")
    stray = sum(len(entry["uncovered_windows_lines"]) for entry in summary["files"].values())
    if stray:
        print(f"uncovered lines that fall inside Windows-only regions: {stray}")


def print_compare(baseline: dict, summary: dict) -> None:
    print(f"\n{'file':24} {'lines':>18} {'branches':>18} {'functions':>18}")
    for name in sorted(set(baseline["files"]) | set(summary["files"])):
        old = baseline["files"].get(name)
        new = summary["files"].get(name)
        cells = []
        for key in ("lines", "branches", "functions"):
            before = old[key]["covered"] if old else 0
            after = new[key]["covered"] if new else 0
            total = new[key]["total"] if new else (old[key]["total"] if old else 0)
            cells.append(f"{before:4}->{after:<4} {percent(after, total)}")
        print(f"{name:24} {cells[0]:>18} {cells[1]:>18} {cells[2]:>18}")
    totals = summary["totals"]
    old_totals = baseline["totals"]
    for key in ("lines", "branches", "functions"):
        before = old_totals[key]["covered"]
        after = totals[key]["covered"]
        total = totals[key]["total"]
        print(f"TOTAL {key:10} {before:4} -> {after:<4} ({percent(after, total)})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, help="write the machine-readable summary here")
    parser.add_argument("--baseline", type=Path, help="compare against an earlier --json-out file")
    parser.add_argument("--quiet", action="store_true", help="print only the table")
    args = parser.parse_args()

    files = collect()
    summary = summarize(files)
    print_table(summary)
    if not args.quiet:
        print_uncovered(summary)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nsummary written to {args.json_out}")
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        print_compare(baseline, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
