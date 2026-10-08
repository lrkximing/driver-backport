# Driver Backport

## Environment

Requires Python 3.11+, `make`, a kernel build toolchain, `libclang`, the target Linux kernel source tree, and the donor kernel source tree matching the selected case's `SourceVersion`. The target and donor `compile_commands` must reflect their actual configurations and architectures. These inputs are not included in this repository.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Set `OPENAI_API_KEY` and `OPENAI_MODEL` through environment variables or a local secret manager. For another Chat Completions-compatible service, set `OPENAI_BASE_URL`. Set `OPENAI_STREAM=true` for streaming responses. Do not commit credentials or `driver_backport/local.model.json`. If Python cannot find `libclang`, set `LIBCLANG_FILE` or `LIBCLANG_PATH`.

## Dataset

[`datasets/driver_cases_v6.6.csv`](datasets/driver_cases_v6.6.csv) contains 92 cases. `Path` is relative to the kernel tree, `SourceVersion` identifies the donor version, and `BaselineOld` identifies the target version. `--rows` uses physical CSV line numbers: line 1 is the header, and the first case is on line 2. The dataset does not include experimental results, kernel sources, or compilation databases.

## Run

From the repository root, generate a single-case configuration without calling the model:

```sh
python -m driver_backport.cli.automate_ast_migration dataset-migrate \
  datasets/driver_cases_v6.6.csv --rows 2 --dry-run \
  --old-root /path/to/linux-6.6 \
  --donor-root /path/to/matching-donor-linux \
  --compile-commands-matrix-root /path/to/compile-matrix \
  --compile-commands-label v6.6 \
  --donor-compile-commands-label donor-v6.13 \
  --output-dir /path/to/case-output
```

Compilation databases are located at `<matrix-root>/cases/compile_commands_dataset_<label>/<case-id>.json`. For another case, select the donor source tree and label matching its `SourceVersion`. Check the generated target architecture, configuration, compiler commands, and build target before running:

```sh
python -m driver_backport.cli.automate_ast_migration auto-migrate \
  /path/to/case-output/cases/dataset-002-rtw8821au.case.json \
  --model "$OPENAI_MODEL" --max-repair-rounds 5
```

Results are written to the case state directory. Check `final_compile_validated` and `semantic_complete` separately: a successful build does not imply semantic closure. For batch options, see `dataset-migrate --help`.
