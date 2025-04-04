# Changelog

## v0.10.0 - 2025/04/04

- Added `--external`, `--no-external`

## v0.9.1 - 2025/03/20

- Fixed issues with plugin symlink resolution

## v0.9.0 - 2025/03/19

- Added support for reading plugins from environment
- Added diagnostics if a specified plugin does not exist
- Fixed plugin changes not trigging session restarts in some circumstances

## v0.8.0 - 2025/03/16

- Added `--plugins`

## v0.7.2 - 2025/03/11

- Fixed some additional compiler flag-related breakages

## v0.7.1 - 2025/02/26

- Fixed some additional compiler flag-related breakages

## v0.7.0 - 2025/02/12

- Added `--fix`
- Fixed clang's `-ftime-trace` causing failures in some cases

## v0.6.0 - 2025/02/10

- Added `--relative-paths` to enable relative paths in output
- Fixed paths in output always being relative by default

## v0.5.2 - 2025/01/30

- Fixed compiler argument slicing regression introduced in v0.5.1

## v0.5.1 - 2025/01/29

- Fixed sanitizer and GCC flags breaking clang-tidy in some common cases

## v0.5.0 - 2025/01/27

- Added `--batch x/y` to enable distributed parallelism
- Minor internal fixes

## v0.4.1 - 2024/11/19

- Fixed precompiled headers breaking runs in some circumstances
- Improved wording of session restart message

## v0.4.0 - 2024/09/07

- Fixed sessions not restarting if `.clang-tidy` is modified
- Added `--labels-only`
- Minor performance improvements

## v0.3.0 - 2024/08/04

- Added `--no-session`
- Added use of sessions by default

## v0.2.0 - 2024/08/03

- Added `--session`

## v0.1.2 - 2024/06/06

- Fixed build-generated translation causing 'did not exist or was not a file' errors
- Improved performance of TU enumeration step

## v0.1.1 - 2024/05/16

- Fixed issues with older clang-tidy versions trying to use `--use-color`

## v0.1.0 - 2024/05/09

- First public release 🎉&#xFE0F;
