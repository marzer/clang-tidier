# clang-tidier

A simple `clang-tidy` runner for C and C++ projects.

## Installation

`clang-tidier` requires Python 3.8 or higher, as well as some version of `clang-tidy` visible on the system PATH.

```
pip3 install clang-tidier
```

## Usage

`clang-tidier` is a command-line application

```
usage: clang-tidier [-h] [--version] [--include <regex> [<regex> ...]] [--exclude <regex> [<regex> ...]]
                    [--werror] [--threads <num>] [--batch num/denom] [--session | --no-session]
                    [--relative-paths | --no-relative-paths] [--fix | --no-fix]
                    [--plugins <path...> [<path...> ...]]
                    [compile_db_path]

clang-tidy runner for C and C++ projects.

positional arguments:
  compile_db_path       path to compile_commands.json, or a directory containing it (default: discover
                        automatically)

options:
  -h, --help            show this help message and exit
  --version             print the version and exit
  --include <regex> [<regex> ...]
                        regular expression to select source files.
  --exclude <regex> [<regex> ...]
                        regular expression to exclude source files.
  --werror              stop on the first file that emits warnings
  --threads <num>       number of threads to use.
  --batch num/denom     batch subdivisions.
  --session, --no-session
                        saves run information so subsequent re-runs may avoid re-scanning files.
  --relative-paths, --no-relative-paths
                        show paths as relative to CWD where possible.
  --fix, --no-fix       attempt to apply clang-tidy fixes where possible.
  --plugins <path...> [<path...> ...]
                        one or more plugins to load.

v0.9.0 - github.com/marzer/clang-tidier
```

## Clang-tidy plugins

Clang tidy plugins can be specified in two ways:

- directly using the argument `--plugins` (`--load` also works for compatibility with `clang-tidy`)
- indirectly using either environment variables `CLANG_TIDY_PLUGINS` or `CLANG_TIDIER_PLUGINS`

Specifying multiple plugins via environment variable requires delimiting with semicolons. Regular unix-style colon
delimiters are also supported on Unix.

Plugins specified on the command-line must exist; the program will exit with an error if they do not. Plugins specified
via environment variable will be ignored with a warning if they are not found.

## Exit codes

| Value                                | Meaning                |
| :----------------------------------- | :--------------------- |
| 0                                    | No issues were found   |
| 1                                    | Issues were found      |
| -1                                   | A fatal error occurred |
