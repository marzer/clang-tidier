#!/usr/bin/env python3
# This file is a part of marzer/clang-tidier and is subject to the the terms of the MIT license.
# Copyright (c) Mark Gillard <mark.gillard@outlook.com.au>
# See https://github.com/marzer/clang-tidier/blob/main/LICENSE.txt for the full license text.
# SPDX-License-Identifier: MIT

import argparse
import multiprocessing
import re
import shutil
import signal
import subprocess
import sys
import misk
import json
import os
import concurrent.futures as futures
from typing import Tuple
from io import StringIO
from pathlib import Path

import colorama

from . import paths
from .colour import *
from .version import *

IS_WORKER = False
STOP = None
FATAL_ERROR = None
SESSION_FILE_LOCK = None
PROBLEMATIC_FILE_COUNT = None
ANSI_ESCAPE_CODES = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')


def error(text):
    print(rf"{bright(rf'error:', 'red')} {text}", file=sys.stderr)


def sigint_handler(signal, frame):
    if STOP is not None:
        STOP.set()
    if not IS_WORKER:  # main thread
        print('\nKeyboardInterrupt caught; aborting...', flush=True)
    else:
        sys.exit(0)


def initialize_worker():
    global IS_WORKER
    IS_WORKER = True
    signal.signal(signal.SIGINT, sigint_handler)


def clean_clang_tidy_output(s: str):
    if s is None:
        return ''

    # fix windows nonsense
    s = str(s).replace("\r\n", "\n")

    # strip away ansi nonsense
    s = ANSI_ESCAPE_CODES.sub('', s)

    # strip "XXXXXX warnings/errors generated."
    s = re.sub(r'[0-9]+ (?:warning|error)s? (?:and [0-9]+ (?:warning|error)s? )?generated[.]?', r'', s)

    # strip "error while processing XXXXXXX"
    s = re.sub(r'[Ee]rror while processing [./a-zA-Z0-9_+-]+[.]', r'', s)

    return s.strip()


def find_upwards(name: str, files=True, directories=False, start_dir: Path = Path.cwd()) -> Path:
    assert name is not None
    assert start_dir is not None
    prev = None
    while True:
        dir = start_dir.resolve() if prev is None else prev.parent
        if not dir or dir == prev:
            break
        candidate = dir / name
        if files and candidate.is_file():
            return candidate
        if directories and candidate.is_dir():
            return candidate
        prev = dir
    return None


def get_relative_path(p: Path, relative_to: Path = Path.cwd()) -> Path:

    p = misk.coerce_path(p).resolve()
    relative_to = misk.coerce_path(relative_to).resolve()
    try:
        return Path(os.path.relpath(str(p), str(relative_to)))
    except:
        return p


def record_file_completed(session_file: Path, source_file: Path):
    global SESSION_FILE_LOCK
    SESSION_FILE_LOCK.acquire()
    try:
        source_file = str(source_file)
        session_file.touch()
        with open(str(session_file), encoding='utf-8', mode='r+') as f:
            session = json.load(f)
            session: dict
            assert isinstance(session, dict)
            if 'sources' not in session:
                session['sources'] = dict()
            if source_file not in session['sources']:
                session['sources'][source_file] = dict()
            session['sources'][source_file]['completed'] = True
            f.seek(0)
            json.dump(session, f, indent="\t")
            f.truncate()
    finally:
        SESSION_FILE_LOCK.release()


def worker(
    clang_tidy_exe: str,
    clang_tidy_version: Tuple[int, int, int],
    compile_db: Path,
    werror: bool,
    src_file: Path,
    session_file: Path,
):
    global STOP
    global FATAL_ERROR
    global PROBLEMATIC_FILE_COUNT
    global SESSION_FILE_LOCK
    if STOP.is_set():
        return

    assert src_file is not None
    try:
        src_file = misk.coerce_path(src_file).resolve()

        proc = subprocess.run(
            [
                clang_tidy_exe,
                rf'-p={compile_db}',
                '--quiet',
                '--warnings-as-errors=-*',  # none
                '--extra-arg=-D__clang_tidy__',
            ]
            + (['--use-color=false'] if clang_tidy_version[0] >= 12 else [])
            + [src_file],
            cwd=str(Path.cwd()),
            encoding='utf-8',
            capture_output=True,
            check=False,
        )

        def find_error(s):
            return re.search(rf'\s?(?:[Ww]arning|[Ee]rror|WARNING|ERROR):\s?', s)

        stdout = clean_clang_tidy_output(proc.stdout)
        stderr = clean_clang_tidy_output(proc.stderr)
        stdout = ('stdout', stdout, find_error(stdout))
        stderr = ('stderr', stderr, find_error(stderr))
        found_error = stdout[2] or stderr[2]
        both_have_content = stdout[1] and stderr[1]
        either_have_content = stdout[1] or stderr[1]
        indent = "    " if both_have_content else ""

        if proc.returncode != 0:
            FATAL_ERROR.set()
        if found_error:
            with PROBLEMATIC_FILE_COUNT.get_lock():
                PROBLEMATIC_FILE_COUNT.value += 1
        if proc.returncode != 0 or found_error:
            if werror:
                STOP.set()
            termw = shutil.get_terminal_size((80, 20))
            termw = min(120, termw.columns)
            msg = ''
            if either_have_content:
                for name, content, _ in (stdout, stderr):
                    if not content:
                        continue
                    if both_have_content:
                        msg += f"\n  {name}:"
                    for line in content.splitlines():
                        if line.strip():
                            s = f"\n{indent}{line}"
                            s = s.replace(
                                "warning:",
                                (bright('error:', colour='RED') if werror else bright('warning:', colour='YELLOW')),
                            )
                            s = re.sub(r'(\[[a-zA-Z0-9.-]+\])$', lambda m: bright(m[1]), s)
                            msg += s
                    msg = msg.replace(str(src_file), bright(get_relative_path(src_file)))
            if proc.returncode != 0:
                msg += f"\nclang-tidy subprocess exited with code {proc.returncode}."
            if msg.startswith('\n'):
                msg = msg[1:]
            print(msg, flush=True)
        else:
            if session_file:
                record_file_completed(session_file, src_file)
            print(f'No problems found in {bright(get_relative_path(src_file))}.', flush=True)

    except Exception as exc:
        STOP.set()
        if not isinstance(exc, KeyboardInterrupt):
            print(rf'[{type(exc).__name__}] {exc}')
            FATAL_ERROR.set()
            raise


def main_impl():
    args = argparse.ArgumentParser(
        description=r'clang-tidy runner for C and C++ projects.',
        epilog=rf'v{VERSION_STRING} - github.com/marzer/clang-tidier',
    )
    args.add_argument(r'--version', action=r'store_true', help=r"print the version and exit", dest=r'print_version')
    args.add_argument(
        r"compile_db",
        type=Path,
        nargs=r'?',
        default=None,
        help="path to compile_commands.json, or a directory containing it (default: discover automatically)",
    )
    args.add_argument(
        r"--include", type=str, nargs='+', metavar=r"<regex>", help=rf"regular expression to select source files."
    )
    args.add_argument(
        r"--exclude", type=str, nargs='+', metavar=r"<regex>", help=rf"regular expression to exclude source files."
    )
    args.add_argument(r'--werror', action=r'store_true', help=r'stop on the first file that emits warnings')
    args.add_argument(r'--where', action=r'store_true', help=argparse.SUPPRESS)
    args.add_argument(
        r"--threads", type=int, metavar=r"<num>", default=os.cpu_count(), help=rf"number of threads to use."
    )
    args.add_argument(
        r'--resumable',
        action=r'store_true',
        help=r"saves run information so subsequent re-runs may avoid re-scanning files.",
    )
    args = args.parse_args()

    if args.print_version:
        print(VERSION_STRING)
        return

    if args.where:
        print(paths.PACKAGE)
        return

    print(rf'{bright("clang-tidier", colour="cyan")} v{VERSION_STRING} - github.com/marzer/clang-tidier')
    global STOP
    STOP = multiprocessing.Event()

    # find compile_commands.json
    if args.compile_db is None:
        # look in cwd
        if (Path.cwd() / 'compile_commands.json').is_file():
            args.compile_db = Path.cwd() / 'compile_commands.json'
        # search upwards
        if args.compile_db is None:
            args.compile_db = find_upwards('compile_commands.json')
        # search one step downwards
        if args.compile_db is None:
            for dir in Path.cwd().iterdir():
                if dir.is_dir() and (dir / 'compile_commands.json').is_file():
                    args.compile_db = dir / 'compile_commands.json'
                    break
        if args.compile_db is not None:
            print(rf"found compilation database {bright(get_relative_path(args.compile_db))}")
        else:
            return rf"could not find {bright('compile_commands.json')}"
    else:
        if args.compile_db.exists() and args.compile_db.is_dir():
            args.compile_db /= 'compile_commands.json'
        if not args.compile_db.is_file():
            return rf"compilation database {bright(args.compile_db)} did not exist or was not a file"

    # compute filters
    if not args.include:
        args.include = []
    if not args.exclude:
        args.exclude = []
    args.exclude.append(r'.*/_deps/.*')
    args.exclude.append(r'^/tmp/.*')
    args.include = [re.compile(s) for s in args.include]
    args.exclude = [re.compile(s) for s in args.exclude]

    # read compilation db
    sources = None
    compile_db_hash = ''
    with open(str(args.compile_db), encoding='utf-8') as f:
        db_text = f.read()
        sources = json.loads(db_text)
        compile_db_hash = misk.sha1(db_text)
    if not isinstance(sources, (list, tuple)):
        return rf"expected array at root of {bright('clang-tidy')}; saw {type(sources).__name__}"
    if not sources:
        print("no work to do.")
        return 0
    sources = misk.remove_duplicates(sources)
    if STOP.is_set():
        return 0

    # enumerate translation units
    for i in range(len(sources)):
        source = sources[i]
        sources[i] = None
        if not isinstance(source, dict):
            return rf"expected source [{i}] as JSON object; saw {type(source).__name__}"
        source: dict
        # read file path
        file = source.get('file', None)
        if file is None:
            return rf"expected source [{i}] to have key {bright('file')}"
        file = Path(file)
        if not file.is_absolute():
            directory = source.get('directory', None)
            if directory is not None:
                directory = Path(directory)
            if directory:
                file = directory / file
        # check if the file exists
        if not (file.exists() and file.is_file()):
            continue
        sources[i] = file
    sources = misk.remove_duplicates(sorted([s for s in sources if s is not None]))

    # apply include and exclude filters
    for i in range(len(sources)):
        source = sources[i]
        sources[i] = None
        # apply include filter
        if args.include:
            include = False
            for filter in args.include:
                if filter.search(str(source)):
                    include = True
                    break
            if not include:
                continue
        # apply exclude filter
        if args.exclude:
            include = True
            for filter in args.exclude:
                if filter.search(str(source)):
                    include = False
                    break
            if not include:
                continue
        sources[i] = source
    sources = [s for s in sources if s is not None]
    if not sources:
        print("no work to do.")
        return 0
    if STOP.is_set():
        return 0

    # detect clang-tidy
    clang_tidy_exe = 'clang-tidy'
    clang_tidy_label = clang_tidy_exe
    clang_tidy_version = (0, 0, 0)
    if not shutil.which('clang-tidy'):
        clang_tidy_exe = None
        for i in range(20, 6, -1):
            if shutil.which(rf'clang-tidy-{i}'):
                clang_tidy_exe = rf'clang-tidy-{i}'
                clang_tidy_label = clang_tidy_exe
                clang_tidy_version = (i, 0, 0)
                break
    if clang_tidy_exe is None:
        return rf"could not detect {bright('clang-tidy')}"

    # query the actual version
    try:
        clang_tidy_version_output = clean_clang_tidy_output(
            subprocess.run(
                [clang_tidy_exe, '--version'], cwd=str(Path.cwd()), encoding='utf-8', capture_output=True
            ).stdout
        )
        clang_tidy_version_output = re.search(
            r'\b[vV]?([0-9]+?)[.]([0-9]+?)(?:[.]([0-9]+?))?\b', clang_tidy_version_output
        )
        if clang_tidy_version_output is not None:
            clang_tidy_version = (
                int(clang_tidy_version_output[1]),
                int(clang_tidy_version_output[2]),
                int(clang_tidy_version_output[3]) if clang_tidy_version_output[3] is not None else 0,
            )
            print(
                rf"detected {bright(rf'clang-tidy v{clang_tidy_version[0]}.{clang_tidy_version[1]}.{clang_tidy_version[2]}')}"
            )
            clang_tidy_label = rf'clang-tidy-{clang_tidy_version[0]}'
    except:
        pass  # a failure here doesn't really matter, it's just for finer-grained version checking
    if STOP.is_set():
        return 0

    # detect git + filter out gitignored files
    if find_upwards(".git", files=False, directories=True, start_dir=args.compile_db.parent) is not None:
        if shutil.which('git') is not None:
            gitignored_sources = set()
            for source in sources:
                if (
                    subprocess.run(
                        ['git', 'check-ignore', '--quiet', str(source)],
                        capture_output=True,
                        encoding='utf-8',
                        cwd=str(Path(source).parent),
                        check=False,
                    ).returncode
                    == 0
                ):
                    gitignored_sources.add(source)
            sources = [s for s in sources if s not in gitignored_sources]
            if not sources:
                print("no work to do.")
                return 0

        else:
            print(
                rf"{bright(rf'warning:', 'yellow')} detected a git repository but could not detect {bright('git')}; .gitignore rules will not be respected"
            )
    if STOP.is_set():
        return 0

    # session
    session_file = None
    session = None
    if args.resumable:
        session_id = misk.sha1(str(args.compile_db.resolve()))
        # session_dir = paths.TEMP
        session_dir = Path.cwd()
        session_dir.mkdir(exist_ok=True)
        session_file = session_dir / rf'session_{session_id}.json'
        write_session = False
        if session_file.exists():
            try:
                with open(session_file, encoding='utf-8') as f:
                    session = json.load(f)
            except:
                print(rf"{bright(rf'warning:', 'yellow')} session could not be resumed: could not read session file")
                try:
                    session_file.unlink()
                except:
                    pass
        if session is None:
            session = dict()
            write_session = True

        def reset_sources():
            nonlocal session
            nonlocal session_id
            session['sources'] = dict()
            write_session = True

        if 'id' not in session or session['id'] != session_id:
            session['id'] = session_id
            reset_sources()
        if 'hash' not in session or session['hash'] != compile_db_hash:
            session['hash'] = compile_db_hash
            reset_sources()
        if 'compile_db' not in session or session['compile_db'] != str(args.compile_db.resolve()):
            session['compile_db'] = str(args.compile_db.resolve())
            reset_sources()
        if 'clang_tidy_version' not in session or tuple(session['clang_tidy_version']) != clang_tidy_version:
            session['clang_tidy_version'] = clang_tidy_version
            reset_sources()
        if 'sources' not in session:
            reset_sources()
        completed_sources = set()
        all_completed = True
        for source in sources:
            source: Path
            source_key = str(source.resolve())
            if source_key not in session['sources']:
                session['sources'][source_key] = dict()
                write_session = True
            source_obj = session['sources'][source_key]
            source_modified = source.stat().st_mtime_ns
            if 'modified' not in source_obj or source_obj['modified'] != source_modified:
                source_obj['modified'] = source_modified
                source_obj['completed'] = False
                write_session = True
            if 'completed' not in source_obj:
                source_obj['completed'] = False
                write_session = True
            if source_obj['completed']:
                completed_sources.add(source)
            else:
                all_completed = False
        if all_completed:
            for _, v in session['sources'].items():
                v['completed'] = False
            completed_sources.clear()
            write_session = True
        if write_session:
            print(rf"{'re' if all_completed else ''}starting session {bright(session_id)}")
            with open(session_file, encoding='utf-8', mode='w') as f:
                json.dump(session, f, indent="\t")
        else:
            print(rf"resuming session {bright(session_id)}")

        sources = [s for s in sources if s not in completed_sources]
        if not sources:
            print("no work to do.")
            return 0
        if STOP.is_set():
            return 0

    # run clang-tidy on each file
    global FATAL_ERROR
    global PROBLEMATIC_FILE_COUNT
    global SESSION_FILE_LOCK
    FATAL_ERROR = multiprocessing.Event()
    PROBLEMATIC_FILE_COUNT = multiprocessing.Value('i', 0)
    SESSION_FILE_LOCK = multiprocessing.Lock()
    print(rf'running {bright(clang_tidy_label)} on {len(sources)} file{"s" if len(sources) > 1 else ""}')
    with futures.ProcessPoolExecutor(
        max_workers=max(min(os.cpu_count(), len(sources), args.threads), 1), initializer=initialize_worker
    ) as executor:
        jobs = [
            executor.submit(worker, clang_tidy_exe, clang_tidy_version, args.compile_db, args.werror, f, session_file)
            for f in sources
        ]
        for future in futures.as_completed(jobs):
            if STOP.is_set():
                future.cancel()
                continue
            try:
                future.result()
            except Exception as exc:
                STOP.set()
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    executor.shutdown(wait=False)
                if not isinstance(exc, KeyboardInterrupt):
                    print(rf'[{type(exc).__name__}] {exc}')
                    FATAL_ERROR.set()
                    raise

    if FATAL_ERROR.is_set():
        return r'An error occurred.'
    with PROBLEMATIC_FILE_COUNT.get_lock():
        if PROBLEMATIC_FILE_COUNT.value:
            print(rf'{bright(clang_tidy_label)} found problems in {PROBLEMATIC_FILE_COUNT.value} file(s).')
            return int(PROBLEMATIC_FILE_COUNT.value)
    return 0


def main():
    signal.signal(signal.SIGINT, sigint_handler)
    colorama.init()
    result = None
    try:
        result = main_impl()
        if result is None:
            sys.exit(0)
        elif isinstance(result, int):
            sys.exit(result)
        elif isinstance(result, str):  # error message
            error(result)
            sys.exit(-1)
        else:
            error('unexpected result type')
            sys.exit(-1)
    except SystemExit as exit:
        raise exit from None
    except argparse.ArgumentError as err:
        error(err)
        sys.exit(-1)
    except BaseException as err:
        with StringIO() as buf:
            buf.write(
                f'\n{dim("*************", "red")}\n\n'
                'You appear to have triggered an internal bug!'
                f'\n{style("Please file an issue at github.com/marzer/clang-tidier/issues")}'
                '\nMany thanks!'
                f'\n\n{dim("*************", "red")}\n\n'
            )
            misk.print_exception(err, include_type=True, include_traceback=True, skip_frames=1, logger=buf)
            buf.write(f'{dim("*************", "red")}\n')
            print(buf.getvalue(), file=sys.stderr)
        sys.exit(-1)


if __name__ == '__main__':
    main()
