# Driver Backport

Driver Backport is a research harness for migrating Linux drivers from a newer
kernel tree to an older one. It projects requirements from the selected driver
and its concrete build context, adapts only demanded providers and integration
relations, then iterates over compiler, modpost, and remaining demand evidence.
The target build and demand-closure check are separate validation conditions.

This repository contains the migration implementation and a 92-case selection
dataset for Linux 6.6. It does **not** include Linux source trees, generated
compilation databases, experiment state, migration results, kernel modules,
credentials, or machine-specific launch scripts. The CSV identifies cases; it
is not a prebuilt or validated experiment environment.

## Prerequisites

- Python 3.11 or newer, `make`, a Linux kernel toolchain, and a matching
  `libclang` shared library. Cross-architecture cases also need the appropriate
  cross-compiler and `ARCH`/`CROSS_COMPILE` build settings.
- A clean checkout of the target Linux kernel and the donor kernel version(s)
  indicated by each selected CSV row's `SourceVersion`.
- Build configurations and real compilation commands for the selected target
  and donor translation units. Compilation databases are inputs tied to those
  exact trees and build contexts; do not reuse one from another kernel version
  or architecture.
- Access to an OpenAI-compatible Chat Completions model endpoint. Set
  `OPENAI_API_KEY`, `OPENAI_MODEL`, and, for a non-default provider,
  `OPENAI_BASE_URL` in your environment or a secret manager. `OPENAI_STREAM`
  controls streaming. Do not commit keys or local model configuration.

Install the Python dependencies in a virtual environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The `clang` Python binding must match the installed `libclang`. If discovery
fails, set `LIBCLANG_FILE` or `LIBCLANG_PATH`. The pinned tree-sitter packages
are the pair used in the project's verified parser environment.

## Dataset

[`datasets/driver_cases_v6.6.csv`](datasets/driver_cases_v6.6.csv) contains
92 unique driver cases. `Path` is relative to the kernel tree;
`SourceVersion` selects the donor release and `BaselineOld` identifies the
intended target release. The dataset does not carry per-case build products or
successful migration claims. Select rows with `--rows` using the CSV's physical
row numbers (including the header; for example, row 2 is the first case).

## Run a case

Run commands from the repository root. First generate a case specification
without calling the model:

```sh
python -m driver_backport.cli.automate_ast_migration dataset-migrate \
  datasets/driver_cases_v6.6.csv \
  --rows 2 --dry-run \
  --old-root /path/to/target-linux \
  --donor-root /path/to/matching-donor-linux \
  --compile-commands-matrix-root /path/to/compile-matrix \
  --compile-commands-label v6.6 \
  --donor-compile-commands-label donor-v6.13 \
  --output-dir /path/to/case-output
```

The example selects CSV row 2, whose donor release is v6.13. For other rows,
use the corresponding donor tree and label. The matrix layout expected by the
case generator is
`cases/compile_commands_dataset_<label>/<case-id>.json` under the supplied
matrix root, for both target and donor labels. These JSON files must contain
actual compilation commands for their respective configured kernel builds;
the generator does not create them. Inspect the generated case JSON and verify
its roots, build configuration, architecture, expected object/module, and
compilation database paths before using API quota.

Then migrate the generated case:

```sh
python -m driver_backport.cli.automate_ast_migration auto-migrate \
  /path/to/case-output/cases/<case-id>.case.json \
  --model "$OPENAI_MODEL" --max-repair-rounds 5
```

The case's state directory contains `migration_result.json` and the evidence
for individual repair rounds. Keep target worktrees and state directories
isolated across parallel cases. A compiled result (`final_compile_validated`)
is not automatically a semantic or functional success; inspect
`semantic_complete`, the stop reason, and the recorded validation evidence.
Kernel-module loading and hardware/fuzz testing require separate setup.

The CLI also supports `dataset-migrate` without `--dry-run` to prepare and run
selected cases. Use `python -m driver_backport.cli.automate_ast_migration
--help` and `dataset-migrate --help` for all configuration and reset options.
Do not use reset flags on a case with work you intend to resume.

## Repository scope

`driver_backport/analysis`, `scope`, and `planning` construct build-conditioned
requirements and repair units; `migration` applies bounded candidate edits;
`validation` checks the exact target build; `llm` handles model transport; and
`cli` orchestrates the workflow. Historical batch launchers, old case state,
private configuration, result archives, and test suites are intentionally not
part of this publication.
