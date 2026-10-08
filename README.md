# Driver Backport

## 环境配置

需要 Python 3.11+、`make`、内核编译工具链、`libclang`、目标 Linux 内核源码，以及与所选案例 `SourceVersion` 对应的 donor 内核源码。目标和 donor 的 `compile_commands` 必须来自各自实际的配置和架构；这些文件不随仓库提供。

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

通过环境变量或本机密钥管理器配置 `OPENAI_API_KEY`、`OPENAI_MODEL`；使用其他兼容 Chat Completions 的服务时设置 `OPENAI_BASE_URL`。如需流式响应，设置 `OPENAI_STREAM=true`。不要将密钥或 `driver_backport/local.model.json` 提交到仓库。若 Python 找不到 `libclang`，设置 `LIBCLANG_FILE` 或 `LIBCLANG_PATH`。

## 数据集

[`datasets/driver_cases_v6.6.csv`](datasets/driver_cases_v6.6.csv) 包含 92 个案例。`Path` 是内核树内的相对路径，`SourceVersion` 是 donor 版本，`BaselineOld` 是目标版本。`--rows` 使用 CSV 的物理行号，首行是表头；第一个案例为第 2 行。数据集不包含实验结果、内核源码或编译数据库。

## 运行

从仓库根目录生成单个案例配置（不调用模型）：

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

编译数据库的路径格式是 `<matrix-root>/cases/compile_commands_dataset_<label>/<case-id>.json`。更换案例时，按其 `SourceVersion` 更换 donor 源码和标签。核对生成的目标架构、配置、编译命令与构建目标后运行：

```sh
python -m driver_backport.cli.automate_ast_migration auto-migrate \
  /path/to/case-output/cases/dataset-002-rtw8821au.case.json \
  --model "$OPENAI_MODEL" --max-repair-rounds 5
```

结果写入案例的状态目录；分别查看 `final_compile_validated` 与 `semantic_complete`，编译通过不等于语义闭合。批量运行选项见 `dataset-migrate --help`。
