import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]

# Inclusion list of targets we wish to analyze.
# Exclusion list of targets within these that we exclude
# is located in pyproject.toml (tool.ruff.exclude).
FORMAT_TARGETS = [
    "tools",
    "dynamic_gsplats",
    "tests",
    "setup.py",
]


def parse_args():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--verbose", action="store_true")
    common.add_argument("--filter-to", nargs="+", required=False,
                        help="Filters processing to only input files, will lint entire repo if not provided")

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    subparsers.add_parser("ruff", parents=[common])
    subparsers.add_parser("mypy", parents=[common])
    subparsers.add_parser("all", parents=[common])

    return parser.parse_args()


def start_execute_in_bg(cmd):
    # Start running a command on the event loop and return
    # the asyncio task.
    return asyncio.create_task(asyncio.create_subprocess_exec(cmd[0], *cmd[1:]))


def get_target_files(directories: List[str] = FORMAT_TARGETS, filter_to: Optional[List[str]] = None) -> List[str]:
    """
    Takes a list of directories and filter_to files and
    returns only files found in the input directories.
    If filter_to is None, returns the full input list of directories.
    """

    # Return all directories as full filepath strings.
    if filter_to is None:
        return [str(ROOT / d) for d in directories]

    # Convert to Paths.
    files: List[Path] = [ROOT / f for f in filter_to]
    dirs: List[Path] = [ROOT / d for d in directories]

    # Check that the file exists. This shouldn't happen due to how this gets called,
    # but better to be robust to upstream errors.
    files = [f for f in files if f.exists()]

    # Filter to only files that are inside the target directories.
    return [str(f) for f in files if any(f.is_relative_to(d) for d in dirs)]


async def main(lint_args):

    failed = False
    tasks = {}
    target_files = get_target_files(filter_to=args.filter_to)

    if lint_args.action in {"ruff", "all"}:
        tasks["ruff"] = start_execute_in_bg(["ruff", "check"]
                                            + target_files)
    if lint_args.action in {"mypy", "all"}:
        tasks["mypy"] = start_execute_in_bg(["mypy", *target_files])

    if "ruff" in tasks:
        code, out, err = await tasks["ruff"]
        if code != 0:
            print(
                "[ruff] Some of your files are not formatted correctly. "
                "Run `ruff check`"
                "Ruff output:\n```\n%s```",
                out + err,
            )
            failed = True
        else:
            print("[ruff] passed.\n%s", out + err)

    if "mypy" in tasks:
        code, out, err = await tasks["mypy"]
        if code != 0:
            print("[mypy] detected some issues with your code:```\n%s```", out + err)
            failed = True
        else:
            print("[mypy] passed.\n%s", out + err)

    if failed:
        raise Exception("Linting failed.")


if __name__ == "__main__":
    SUCCESS = True
    try:
        args = parse_args()
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("Keyboard interrupt ...")
        SUCCESS = False
    except Exception as e:  # pylint: disable=broad-except
        print("Exception happened %r", e)
        print(e, file=sys.stderr)
        SUCCESS = False

    if not SUCCESS:
        sys.exit(-1)
