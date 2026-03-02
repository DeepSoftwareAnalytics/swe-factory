# Inference

Two-stage pipeline for running editing agents on SWE environments:

```
Stage 1: build/transfer images  ->  Stage 2: run agents
```

## Setup

```bash
conda create -n inference python=3.13 -y
conda activate inference
pip install -r requirements-inference.txt
```

## Stage 1: Build SWE Env Images Locally

### Motivation

After data collection, we typically have Dockerfiles for each instance. To run
experiments, we build these images locally first. This stage also normalizes
the environment so downstream evaluation is stable and less noisy. Common
issues we handle here include:
- Reward/test execution taking too long (e.g. >300s).
- Repositories not mounted at `/testbed` (e.g., code lives under `/testbed/mypy`).
- Extra environment files under `/testbed` causing noisy diffs/patches (e.g., `/testbed/.venv`).

### Input

Use the transfer agent runner (LLM-driven) to normalize the environment and build images,
then embed artifacts (Dockerfile + eval script) into a new dataset file. Use the
dataset produced by the SWE-Builder stage (the output of `app/main.py` under the
`results` directory) as-is.

### LLM Configuration (Stage 1)

> [!IMPORTANT]
> Stage 1 uses direct OpenAI-compatible chat completions (not LiteLLM).

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_BASE_URL="YOUR_URL"  # optional override
```

Set the model name via `--model_name <model_name>` (required).

### Command

```bash
python inference/build_image/main.py \
  --input /path/to/instances.json \
  --output /path/to/run_dir \
  --max-iterations 5 \
  --eval-timeout 300 \
  --max-workers 2 \
  --model_name <model_name>
```

### Notes

- `--eval-timeout` defaults to 300 seconds if omitted. Instances that exceed the timeout are marked
  as failures and filtered out of the transferred dataset.

### Parameters (Stage 1)

| Parameter | Meaning | Allowed / Example |
| --- | --- | --- |
| `--input` | Path to raw instances JSON list | `/path/to/instances.json` |
| `--output` | Output directory for build artifacts | `/path/to/run_dir` |
| `--max-iterations` | LLM edit/build iterations per instance | `5` |
| `--eval-timeout` | Eval script timeout (seconds) | `300` |
| `--max-workers` | Parallel workers | `2` |
| `--skip-existing` | Skip instances with existing `summary.json` | flag |
| `--model_name` | LLM model name | `<model_name>` |

### Outputs

Outputs in `--output`:
- `summary.json` / `summary_main.json`
- `<input_stem>_transferred.json` (successful entries with `docker_image`, `dockerfile`, `eval_script`).
  The `docker_image` field points to the built SWE environment image for each instance.
- `<input_stem>_failed.json`

## Stage 2: Run a Coding Agent on the Built SWE Environment

### Supported Agents

Run against the transferred dataset produced in Stage 1.

| Agent | Scaffold | Tools | Notes | Recommended Use |
| --- | --- | --- | --- | --- |
| mini_swe_agent | `mini_swe_agent` | bash-only | non-fn only | multi-language / non-Python repos |
| live_swe_agent | `live_swe_agent` | bash-only | non-fn only | multi-language / non-Python repos |
| DeepSWE (r2egym) | `r2egym` | Python tools | fn + non-fn supported | Python repos |
| OpenHands (experimental, unofficial) | `openhands` | Python tools | fn + non-fn supported | Python repos |

OpenHands diverges significantly from the original implementation, so please use it with caution.

### Dataset Requirements

The dataset should be the Stage 1 transferred output (for example,
`/path/to/run_dir/<input_stem>_transferred.json`). Each entry should include:
`instance_id`, `docker_image`, `dockerfile`, and `eval_script`.

Model calls go through LiteLLM. Set your base URL and provider API key before
running (examples below).

### LLM Configuration (Stage 2)

```bash
export LLM_BASE_URL="YOUR_URL"
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY"
```

### Examples

> [!NOTE]
> Run these commands from the repo root. If you run from another directory,
> set `PYTHONPATH` to the repo path first.

mini_swe_agent:
```bash
python -m inference.agenthub.run.edit runagent_multiple \
  --dataset /path/to/TRANSFERRED_DATASET.json \
  --split dev \
  --k 1 \
  --start_idx 0 \
  --max_workers 5 \
  --traj_dir ./run_logs/mini_swe_run \
  --exp_name mini_swe_run \
  --llm_name openai/gpt-4o-mini \
  --use_fn_calling False \
  --backend docker \
  --scaffold mini_swe_agent
```

DeepSWE editing agent (r2egym scaffold):
```bash
python -m inference.agenthub.run.edit runagent_multiple \
  --dataset /path/to/TRANSFERRED_DATASET.json \
  --split dev \
  --k 1 \
  --start_idx 0 \
  --max_workers 5 \
  --traj_dir ./run_logs/deepswe_run \
  --exp_name deepswe_run \
  --llm_name openai/gpt-4o-mini \
  --use_fn_calling True \
  --backend docker \
  --scaffold r2egym
```

live_swe_agent:
```bash
python -m inference.agenthub.run.edit runagent_multiple \
  --dataset /path/to/TRANSFERRED_DATASET.json \
  --split dev \
  --k 1 \
  --start_idx 0 \
  --max_workers 5 \
  --traj_dir ./run_logs/live_swe_run \
  --exp_name live_swe_run \
  --llm_name openai/gpt-4o-mini \
  --use_fn_calling False \
  --backend docker \
  --scaffold live_swe_agent
```

OpenHands (experimental, unofficial):
```bash
python -m inference.agenthub.run.edit runagent_multiple \
  --dataset /path/to/TRANSFERRED_DATASET.json \
  --split dev \
  --k 1 \
  --start_idx 0 \
  --max_workers 5 \
  --traj_dir ./run_logs/openhands_run \
  --exp_name openhands_run \
  --llm_name openai/gpt-4o-mini \
  --use_fn_calling True \
  --backend docker \
  --scaffold openhands
```

Notes:
- `--split` is only a label for the local JSON loader; keep it consistent (e.g., `dev`).
- If you already have local images, you can skip Stage 1 and provide a dataset that
  includes `instance_id`, `docker_image`, `dockerfile`, and `eval_script`.
- For local swefactory inference, ensure your `docker_image` name contains
  `swefactory` so the runtime selects swefactory mode.
- For `r2egym` and `openhands`, you can use either `--use_fn_calling True` or `False`.
  Use `True` only if your model/provider returns tool calls; there is no auto fallback.
- `--backend` currently supports `docker` only.
- If you use a non-OpenAI provider, set the matching API key env var (for example, `ANTHROPIC_API_KEY`).
- This codebase is built on top of R2E-Gym; thanks to the original authors.

### Run Outputs (Stage 2)

Each run writes artifacts to `--traj_dir`. For example:
`./run_logs/my_run`

Directory layout:
```
run_logs/<exp_name>/
  <exp_name>.jsonl
  trajectories.jsonl
  trajectories_rejection_sampling.jsonl
  reward_summary.json
  <instance_id>/
    agent.log
    output_patch.diff
    test_output.log
    metadata.json
```

History-only files:
- `trajectories.jsonl` and `trajectories_rejection_sampling.jsonl`
  - non-fn-calling: each line is a raw `messages` list
  - fn-calling: each line is `{"messages": [...], "tools": [...]}` (tools schema matches the model call)

### Parameters (Stage 2)

| Parameter | Meaning | Allowed / Example |
| --- | --- | --- |
| `--dataset` | Path to Stage 1 transferred dataset | `/path/to/TRANSFERRED_DATASET.json` |
| `--split` | Label for local JSON loader | `dev` |
| `--k` | Number of instances to run | `1` |
| `--start_idx` | Start index in dataset | `0` |
| `--max_workers` | Parallel workers | `5` |
| `--traj_dir` | Output directory for logs/artifacts | `./run_logs/my_run` |
| `--exp_name` | Experiment name | `my_run` |
| `--llm_name` | LiteLLM model name | `openai/gpt-4o-mini` |
| `--use_fn_calling` | Function-calling mode | `True` or `False` (depends on scaffold + model support) |
| `--backend` | Runtime backend | `docker` |
| `--scaffold` | Agent scaffold | `mini_swe_agent` / `r2egym` / `live_swe_agent` / `openhands` |

## Harbor / Terminal-Bench Agents

Besides the built-in `agenthub` scaffolds above, you can also evaluate
Terminal-Bench style agents with Harbor on SWE-Factory environments.

### Install

```bash
pip install harbor
```

### Workflow

Harbor does not read the SWE-Factory dataset directly. First convert your
transferred dataset into Harbor task format, then run Harbor on the converted
task directory.

### Convert SWE-Factory Data to Harbor Format

```bash
python inference/build_image/convert_to_harbor_format.py \
  --input /path/to/TRANSFERRED_DATASET.json \
  --out-dir ./harbor
```

This generates a Harbor-compatible task directory like:

```text
./harbor/
  <task_id>/
    instruction.md
    task.toml
    environment/Dockerfile
    tests/test.sh
    tests/eval.sh
    solution/solve.sh
```

### Run Harbor

```bash
export OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY"

harbor run \
  -p "$(pwd)/harbor" \
  -a terminus-2 \
  -m "openrouter/stepfun/step-3.5-flash" \
  -n 4 \
  -l 10
```

Notes:
- Harbor support is a separate evaluation path from `agenthub`; it is mainly for
  running Terminal-Bench compatible agents.
- Conversion is required before running Harbor.
- Harbor consumes the same Stage 1 transferred dataset format used by Stage 2.

## Acknowledgements

The inference/agenthub module is developed on top of R2E-Gym. Thanks to the
original authors for their work.
