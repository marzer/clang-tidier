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
                    [--werror] [--threads <num>] [--session | --no-session] [compile_db]

clang-tidy runner for C and C++ projects.

positional arguments:
  compile_db            path to compile_commands.json, or a directory containing it (default: discover automatically)

options:
  -h, --help            show this help message and exit
  --version             print the version and exit
  --include <regex> [<regex> ...]
                        regular expression to select source files.
  --exclude <regex> [<regex> ...]
                        regular expression to exclude source files.
  --werror              stop on the first file that emits warnings
  --threads <num>       number of threads to use.
  --session, --no-session
                        saves run information so subsequent re-runs may avoid re-scanning files. (default: True)

v0.3.0 - github.com/marzer/clang-tidier
```

## Exit codes

| Value                                | Meaning                |
| :----------------------------------- | :--------------------- |
| 0                                    | No issues were found   |
| `N`, where `N` is a positive integer | `N` issues were found  |
| -1                                   | A fatal error occurred |

