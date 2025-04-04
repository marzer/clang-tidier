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
import platform
import json
import os
import concurrent.futures as futures
from typing import Tuple, List
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


def normalize_path(p: Path, relative_to: Path = Path.cwd(), relative=False) -> Path:

    p = misk.coerce_path(p).resolve()
    if not relative:
        return p
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
    labels_only: bool,
    relative_paths: bool,
    fix: bool,
    plugins: List[List[str]],
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
                rf'-p={compile_db.parent}',
                '--quiet',
                '--warnings-as-errors=-*',  # none
                '--extra-arg=-D__clang_tidy__',
            ]
            + (['--use-color=false'] if clang_tidy_version[0] >= 12 else [])
            + (['--fix'] if fix else [])
            + ([rf'--load={p[0]}' for p in plugins])
            + [src_file],
            cwd=str(Path.cwd()),
            encoding='utf-8',
            capture_output=True,
            check=False,
        )

        def find_error(s) -> bool:
            return bool(re.search(rf'\s?(?:[Ww]arning|[Ee]rror|WARNING|ERROR):\s?', s))

        stdout = clean_clang_tidy_output(proc.stdout)
        stderr = clean_clang_tidy_output(proc.stderr)
        stdout = ['stdout', stdout, find_error(stdout)]
        stderr = ['stderr', stderr, find_error(stderr)]

        if fix and proc.returncode == 0 and stdout[2] and not stderr[2]:
            m = re.fullmatch(r'clang-tidy\s+applied\s+([0-9]+)\s+of\s+([0-9]+)\s+(suggested\s+)?fixes[.]?', stderr[1])
            if m and m[1] == m[2]:
                stdout[2] = False

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
            msg = ''
            if either_have_content:
                for name, content, _ in (stdout, stderr):
                    if not content:
                        continue
                    if both_have_content:
                        msg += f"\n  {name}:"
                    for line in content.splitlines():
                        if not line.strip():
                            continue
                        if labels_only and (re.match(r'^\s+', line) or re.match(r'^(.+?: )?note:', line)):
                            continue
                        s = f"\n{indent}{line}"
                        msg += s

                    if labels_only:
                        msg = re.sub(
                            r'^.+?: warning: .+? \[([a-zA-Z0-9.,-]+)\]$',
                            lambda m: "\n".join([rf'[{x}]' for x in m[1].split(',')]),
                            msg,
                            flags=re.MULTILINE,
                        )
                        msg = re.sub(r'^\s+?.*?$', '', msg, flags=re.MULTILINE)
                        msg = re.sub(r'\n\n+', '\n', msg)
                        msg = [x.strip() for x in msg.split()]
                        msg = "\n".join(sorted(misk.remove_duplicates([x for x in msg if x])))
                    else:
                        msg = re.sub(
                            r'^(.+?): warning: (.+?) (\[[a-zA-Z0-9.,-]+\])$',
                            lambda m: rf'{m[1]}: {"error" if werror else "warning"}: {m[2]} {bright(m[3])}',
                            msg,
                            flags=re.MULTILINE,
                        )
                        msg = msg.replace('error:', bright('error:', colour='RED'))
                        msg = msg.replace('warning:', bright('warning:', colour='YELLOW'))
                        msg = msg.replace('note:', bright('note:', colour='CYAN'))
                        msg = msg.replace(str(src_file), bright(normalize_path(src_file, relative=relative_paths)))
            if proc.returncode != 0:
                msg += f"\nclang-tidy subprocess exited with code {proc.returncode}."
            if msg.startswith('\n'):
                msg = msg[1:]
            print(msg, flush=True)
        else:
            if session_file:
                record_file_completed(session_file, src_file)
            if not labels_only:
                print(f'No problems found in {bright(normalize_path(src_file, relative=relative_paths))}.', flush=True)

    except Exception as exc:
        STOP.set()
        if not isinstance(exc, KeyboardInterrupt):
            print(rf'[{type(exc).__name__}] {exc}')
            FATAL_ERROR.set()
            raise


def make_boolean_optional_arg(args: argparse.ArgumentParser, name: str, default, help='', **kwargs):
    name = name.strip().lstrip('-')
    if sys.version_info >= (3, 9):
        args.add_argument(rf'--{name}', default=default, help=help, action=argparse.BooleanOptionalAction, **kwargs)
    else:
        dest = name.replace(r'-', r'_')
        args.add_argument(rf'--{name}', action=r'store_true', help=help, dest=dest, default=default, **kwargs)
        args.add_argument(
            rf'--no-{name}',
            action=r'store_false',
            help=(help if help == argparse.SUPPRESS else None),
            dest=dest,
            default=default,
            **kwargs,
        )


def main_impl():
    args = argparse.ArgumentParser(
        description=r'clang-tidy runner for C and C++ projects.',
        epilog=rf'v{VERSION_STRING} - github.com/marzer/clang-tidier',
    )
    args.add_argument(r'--version', action=r'store_true', help=r"print the version and exit", dest=r'print_version')
    args.add_argument(
        r"compile_db_path",
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
    args.add_argument(
        r"--threads", type=int, metavar=r"<num>", default=os.cpu_count(), help=rf"number of threads to use."
    )
    args.add_argument(r"--batch", type=str, metavar=r"num/denom", default="1/1", help=rf"batch subdivisions.")
    make_boolean_optional_arg(
        args, r'session', default=True, help=r'saves run information so subsequent re-runs may avoid re-scanning files.'
    )
    make_boolean_optional_arg(
        args, r'relative-paths', default=False, help=r'show paths as relative to CWD where possible.'
    )
    make_boolean_optional_arg(args, r'external', default=False, help=r'include sources from external/system locations.')
    make_boolean_optional_arg(args, r'fix', default=False, help=r'attempt to apply clang-tidy fixes where possible.')
    args.add_argument(r"--plugins", type=str, nargs='+', metavar=r"<path...>", help=rf"one or more plugins to load.")
    args.add_argument(r"--plugin", type=str, nargs='+', help=argparse.SUPPRESS)
    args.add_argument(r"--load", type=str, nargs='+', help=argparse.SUPPRESS)
    args.add_argument(r'--where', action=r'store_true', help=argparse.SUPPRESS)
    args.add_argument(r'--labels-only', action=r'store_true', help=argparse.SUPPRESS)
    args = args.parse_args()

    if args.print_version:
        print(VERSION_STRING)
        return

    if args.where:
        print(paths.PACKAGE)
        return

    if not args.labels_only:
        print(rf'{bright("clang-tidier", colour="cyan")} v{VERSION_STRING} - github.com/marzer/clang-tidier')
    global STOP
    STOP = multiprocessing.Event()

    # parse batch number
    args.batch = args.batch.strip() if args.batch is not None else ''
    args.batch = args.batch if args.batch else '1/1'
    m = re.fullmatch(r'\s*([+-]?[0-9]+)[/\\: \t.,-]+?([+-]?[0-9]+)\s*', args.batch)
    if not m:
        return rf"batch: could not parse batch information from '{bright(args.batch)}'"
    args.batch = (int(m[1]), int(m[2]))
    if args.batch[0] <= 0 or args.batch[1] <= 0:
        return rf"batch: values must be positive integers"
    if args.batch[0] > args.batch[1]:
        return rf"batch: index must not be greater than count"

    # find compile_commands.json
    if args.compile_db_path is None:
        # look in cwd
        if (Path.cwd() / 'compile_commands.json').is_file():
            args.compile_db_path = Path.cwd() / 'compile_commands.json'
        # search one step downwards
        if args.compile_db_path is None:
            for dir in Path.cwd().iterdir():
                if dir.is_dir() and (dir / 'compile_commands.json').is_file():
                    args.compile_db_path = dir / 'compile_commands.json'
                    break
        # search upwards
        if args.compile_db_path is None:
            args.compile_db_path = find_upwards('compile_commands.json')
        if args.compile_db_path is not None:
            if not args.labels_only:
                print(
                    rf"found compilation database {bright(normalize_path(args.compile_db_path, relative=args.relative_paths))}"
                )
        else:
            return rf"could not find {bright('compile_commands.json')}"
    else:
        if args.compile_db_path.exists() and args.compile_db_path.is_dir():
            args.compile_db_path /= 'compile_commands.json'
        if not args.compile_db_path.is_file():
            return rf"compilation database {bright(args.compile_db_path)} did not exist or was not a file"

    # read compilation db
    compile_db = None
    with open(str(args.compile_db_path), encoding='utf-8') as f:
        db_text = f.read()
        compile_db = json.loads(db_text)
    if not isinstance(compile_db, (list, tuple)):
        return rf"expected array at root of {bright('clang-tidy')}; saw {type(compile_db).__name__}"
    if not compile_db:
        print("no work to do.")
        return 0
    compile_db = misk.remove_duplicates(compile_db)
    if STOP.is_set():
        return 0

    # enumerate translation units
    sources = []
    invalid_pchs = set()
    for i in range(len(compile_db)):
        source = compile_db[i]
        if not isinstance(source, dict):
            return rf"expected source [{i}] as JSON object; saw {type(source).__name__}"
        source: dict
        # sanity-check file path
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
        file = file.resolve()
        # filter out various problematic things
        excluded = False
        if not args.external:
            for exclude_pattern in (r'^/tmp/', r'^/var/tmp/', r'.*[/\\]_deps[/\\].*'):
                if re.search(exclude_pattern, str(file)):
                    excluded = True
                    break
        if excluded:
            continue
        # check if the file exists
        if not (file.exists() and file.is_file()):
            print(rf"{bright(rf'warning:', 'yellow')} source '{file}' did not exist or was not a file; ignoring")
            continue
        # sanity-check command
        command = source.get('command', None)
        if command is None:
            return rf"expected source '{file}' to have key {bright('command')}"
        command = str(command).strip()
        # massage CMake PCH into behaving
        include_pch = re.search(
            r'-Xclang\s+-include-pch\s+-Xclang\s+([^\s]*?cmake_pch.h(?:xx|pp|\+\+|h)?.pch)', command
        )
        if include_pch:
            pch_path = Path(include_pch[1])
            if pch_path in invalid_pchs or not (pch_path.exists() and pch_path.is_file()):
                invalid_pchs.add(pch_path)
                command = command[: include_pch.start()] + command[include_pch.end() :]
        # remove warning flags and other args that muck things up (e.g. GCC flags clang doesn't understand)
        UNWANTED_ARGS = (
            r'-Wl,[a-zA-Z0-9_+=-]+',
            r'-fsanitize(=[a-zA-Z0-9_+-]+)?',
            r'-f(no-)?(time-trace|pch-(instantiate-templates|debuginfo|codegen|preprocess|validate-input-files-content))[a-zA-Z0-9_+=-]*',
            r'-static-asan[a-zA-Z0-9_+=-]*',
            r'-g(gdb[0-9]?|btf|dwarf)[a-zA-Z0-9_+=-]*',
            r'-s',
            r'-W(no-)?(error=)?[a-z][a-zA-Z0-9_+-]*',
        )
        command += ' '
        for arg in UNWANTED_ARGS:
            command = re.sub(rf'\s+{arg}\b', ' ', command)
        # commit back to db
        source['command'] = command.strip()
        source['file'] = str(file)
        sources.append(file)
    compile_db.sort(key=lambda x: x["file"])
    sources = misk.remove_duplicates(sorted([s for s in sources if s is not None]))

    # apply batching
    sources_ = []
    for i in range(args.batch[0] - 1, len(sources), args.batch[1]):
        sources_.append(sources[i])
    sources = sources_

    if not sources:
        print("no work to do.")
        return 0

    # at this point sources[] contains the absolute paths of all the non-temp, non-_deps source files that actually
    # existed, with the file paths of compile_db being synchronized to match.
    #
    # we now prune compile_db of any entries that did not survive this first pass,
    # and generate the compile db hash based on that version of it
    #
    # (note: this is done before applying the user's filters; this is intentional)
    def prune_compile_db_to_match_sources():
        nonlocal compile_db
        nonlocal sources
        compile_db = [x for x in compile_db if Path(x["file"]) in sources]

    prune_compile_db_to_match_sources()
    compile_db_hash = misk.sha1(json.dumps(compile_db, indent="\t"))

    # apply include and exclude filters
    if not args.include:
        args.include = []
    if not args.exclude:
        args.exclude = []
    args.include = [re.compile(s) for s in args.include]
    args.exclude = [re.compile(s) for s in args.exclude]
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
    if STOP.is_set():
        return 0

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
            if not args.labels_only:
                print(
                    rf"detected {bright(rf'clang-tidy v{clang_tidy_version[0]}.{clang_tidy_version[1]}.{clang_tidy_version[2]}')}"
                )
            clang_tidy_label = rf'clang-tidy-{clang_tidy_version[0]}'
    except:
        pass  # a failure here doesn't really matter, it's just for finer-grained version checking
    if STOP.is_set():
        return 0

    # plugins:
    plugins = []
    if True:

        # helper for sanity-checking plugins
        def process_plugins(plugins_in: List[str], from_env=False):
            plugins = [str(p).strip() for p in plugins_in]
            plugins = [p for p in plugins if p]
            plugins = misk.remove_duplicates(sorted(plugins))

            # sanity-check and resolve symlinks
            for i in range(len(plugins)):
                elem = plugins[i]
                path = Path(os.path.abspath(elem))  # normalize without resolving
                target_path = path.resolve()
                if not target_path.is_file():
                    err = rf"plugin {bright(elem)} "
                    if from_env:
                        err += "(from env) "
                    if target_path != path:
                        err += rf"was symbolic link to {bright(target_path)} which "
                    err += rf"did not exist or was not a file"
                    if from_env:
                        print(rf"{bright(rf'warning:', 'yellow')} {err}; it will be ignored")
                        plugins[i] = None
                        continue
                    else:
                        return err
                plugins[i] = target_path

            plugins = [p for p in plugins if p]
            plugins = misk.remove_duplicates(sorted(plugins))

            plugins_in.clear()
            plugins_in += plugins
            return True

        # enumerate plugins from command-line
        plugins = (
            [p for p in (args.plugins if args.plugins else [])]
            + [p for p in (args.plugin if args.plugin else [])]
            + [p for p in (args.load if args.load else [])]
        )
        plugins_ok = process_plugins(plugins)
        if isinstance(plugins_ok, str):
            return plugins_ok

        # enumerate plugins from env
        env_plugins = []
        for key in ('CLANG_TIDY_PLUGINS', 'CLANG_TIDIER_PLUGINS'):
            e = str(os.getenv(key, ''))
            e = re.split(r'[;' + (r'' if (platform.system() == r'Windows' or os.name == r'nt') else r':') + r']+', e)
            e = [p.strip() for p in e]
            e = [p for p in e if p]
            env_plugins += e
        process_plugins(env_plugins, True)
        env_plugins = [p for p in env_plugins if p not in plugins]

        if plugins or env_plugins:
            print(rf"plugins:")
            if plugins:
                print("  " + "\n  ".join([rf"{p}" for p in plugins]))
            if env_plugins:
                print("  " + "\n  ".join([rf"{p} (from env)" for p in env_plugins]))

        # merge both, listify and get the last changed date/time for session purposes
        plugins = plugins + env_plugins
        plugins = misk.remove_duplicates(sorted(plugins))
        plugins = [[str(p), p.stat().st_mtime_ns] for p in plugins]

    # detect git + filter out gitignored files
    if find_upwards(".git", files=False, directories=True, start_dir=args.compile_db_path.parent) is not None:
        if shutil.which('git') is not None:
            ignored_sources = set()
            ignored_dirs = set()
            ok_dirs = set()
            for source in sources:
                # check if it's in a known ignored directory first (cheaper)
                dir = source.parent
                in_ignored_dir = dir in ignored_dirs
                if not in_ignored_dir and dir not in ok_dirs:
                    if (
                        subprocess.run(
                            ['git', 'check-ignore', '--quiet', str(dir)],
                            capture_output=True,
                            encoding='utf-8',
                            cwd=str(dir),
                            check=False,
                        ).returncode
                        == 0
                    ):
                        ignored_dirs.add(dir)
                        in_ignored_dir = True
                    else:
                        ok_dirs.add(dir)
                # now check file
                if not in_ignored_dir and (
                    subprocess.run(
                        ['git', 'check-ignore', '--quiet', str(source)],
                        capture_output=True,
                        encoding='utf-8',
                        cwd=str(dir),
                        check=False,
                    ).returncode
                    == 0
                ):
                    ignored_sources.add(source)
            sources = [s for s in sources if s not in ignored_sources]
            if not sources:
                print("no work to do.")
                return 0
        else:
            print(
                rf"{bright(rf'warning:', 'yellow')} detected a git repository but could not detect {bright('git')}; .gitignore rules will not be respected"
            )
    if STOP.is_set():
        return 0

    # prune compile db and write temp copy
    prune_compile_db_to_match_sources()
    compile_db_id = misk.sha1(str(args.compile_db_path.resolve()))
    compile_db_tmp_path = paths.TEMP / 'compile_db' / rf'{compile_db_id}' / 'compile_commands.json'
    compile_db_tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(compile_db_tmp_path, encoding='utf-8', mode='w') as f:
        json.dump(compile_db, f, indent="\t")

    def delete_temp_compile_db():
        nonlocal compile_db_tmp_path
        try:
            misk.delete_file(compile_db_tmp_path)
            misk.delete_directory(compile_db_tmp_path.parent)
        except:
            pass

    if invalid_pchs:
        print(
            rf"{bright(rf'warning:', 'yellow')} detected precompiled headers with missing compilands; analysis will work but may be incomplete (run the regular build at least once to avoid this)"
        )

    # session
    session_id = compile_db_id
    session_file = paths.TEMP / rf'session_{session_id}.json'
    session = None
    if args.session:
        session_file.parent.mkdir(exist_ok=True)
        write_session = False
        session_existed = False
        if session_file.exists():
            session_existed = True
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

        session_was_reset = False
        session_reset_reason = ''

        def reset_session(reason: str):
            nonlocal session
            nonlocal session_was_reset
            nonlocal session_reset_reason
            nonlocal write_session
            if session_was_reset:
                return
            session['sources'] = dict()
            write_session = True
            session_was_reset = True
            session_reset_reason = str(reason).strip()

        if 'id' not in session or session['id'] != session_id:
            session['id'] = session_id
            reset_session('session ID mismatched')

        if 'location' not in session or session['location'] != str(args.compile_db_path.resolve()):
            session['location'] = str(args.compile_db_path.resolve())
            reset_session('compilation database changed')

        if 'hash' not in session or session['hash'] != compile_db_hash:
            session['hash'] = compile_db_hash
            reset_session('compilation database changed')

        if 'version' not in session or tuple(session['version']) != VERSION:
            session['version'] = VERSION
            reset_session('clang-tidier version changed')

        if 'clang_tidy_version' not in session or tuple(session['clang_tidy_version']) != clang_tidy_version:
            session['clang_tidy_version'] = clang_tidy_version
            reset_session('clang-tidy version changed')

        if 'sources' not in session:
            reset_session('no previously discovered sources')

        if 'plugins' not in session or session['plugins'] != plugins:
            session['plugins'] = plugins
            reset_session('plugins changed')

        # .clang-tidy configs
        config_search_dirs = set()
        for source in sources:
            config_search_dirs.add(source.parent)
        config_modified = set()
        for dir in config_search_dirs:
            cfg = find_upwards(".clang-tidy", files=True, directories=False, start_dir=dir)
            if cfg is not None:
                config_modified.add(cfg)
        config_modified = sorted([x.stat().st_mtime_ns for x in config_modified])
        config_modified = config_modified[-1] if config_modified else 0
        if 'config_modified' not in session or session['config_modified'] != config_modified:
            session['config_modified'] = config_modified
            reset_session('.clang-tidy config changed')

        # other misc build generator files
        build_scripts_modified = []
        for filename in ('build.ninja', 'CMakeCache.txt'):
            file = args.compile_db_path.parent / filename
            if file.exists():
                build_scripts_modified.append(file)
        build_scripts_modified = sorted([x.stat().st_mtime_ns for x in build_scripts_modified])
        build_scripts_modified = build_scripts_modified[-1] if build_scripts_modified else 0
        if 'build_scripts_modified' not in session or session['build_scripts_modified'] != build_scripts_modified:
            session['build_scripts_modified'] = build_scripts_modified
            reset_session('build scripts changed')

        completed_sources = set()
        any_completed = False
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
                any_completed = True
            else:
                all_completed = False
        if all_completed:
            for _, v in session['sources'].items():
                v['completed'] = False
            completed_sources.clear()
            write_session = True
        if write_session:
            with open(session_file, encoding='utf-8', mode='w') as f:
                json.dump(session, f, indent="\t")

        if not args.labels_only:
            if session_existed and (all_completed or session_was_reset):
                print(
                    rf'restarting session {bright(session_id)}{rf" (restarted because {session_reset_reason})" if session_was_reset and session_reset_reason else ""}'
                )
            elif session_existed and any_completed:
                print(rf'resuming session {bright(session_id)}')
            else:
                print(rf'starting session {bright(session_id)}')

        sources = [s for s in sources if s not in completed_sources]
        if not sources:
            print("no work to do.")
            delete_temp_compile_db()
            return 0
    else:
        try:
            session_file.unlink()
        except:
            pass
    if STOP.is_set():
        delete_temp_compile_db()
        return 0

    # run clang-tidy on each file
    global FATAL_ERROR
    global PROBLEMATIC_FILE_COUNT
    global SESSION_FILE_LOCK
    FATAL_ERROR = multiprocessing.Event()
    PROBLEMATIC_FILE_COUNT = multiprocessing.Value('i', 0)
    SESSION_FILE_LOCK = multiprocessing.Lock()
    if not args.labels_only:
        print(rf'running {bright(clang_tidy_label)} on {len(sources)} file{"s" if len(sources) > 1 else ""}')
    with futures.ProcessPoolExecutor(
        max_workers=max(min(os.cpu_count(), len(sources), args.threads), 1), initializer=initialize_worker
    ) as executor:
        jobs = [
            executor.submit(
                worker,
                clang_tidy_exe,
                clang_tidy_version,
                compile_db_tmp_path,
                args.werror,
                f,
                session_file if session is not None else None,
                args.labels_only,
                args.relative_paths,
                args.fix,
                plugins,
            )
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
                    delete_temp_compile_db()
                    raise

    if FATAL_ERROR.is_set():
        delete_temp_compile_db()
        return r'An error occurred.'

    with PROBLEMATIC_FILE_COUNT.get_lock():
        if PROBLEMATIC_FILE_COUNT.value:
            if not args.labels_only:
                print(rf'{bright(clang_tidy_label)} found problems in {PROBLEMATIC_FILE_COUNT.value} file(s).')
            delete_temp_compile_db()
            return 1

    delete_temp_compile_db()
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
