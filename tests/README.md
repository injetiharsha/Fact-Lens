# tests

Automated test suites.

## Structure
- `unit/`: module-level tests.
- `integration/`: end-to-end/API integration tests.
- `llm_toggle/`: LLM ON/OFF comparison traces and benchmark outputs.

Run tests after pipeline/config/model changes.

## Current Benchmark Traces

- `tests/llm_toggle/venv_gpu_trace_rerun/trace_run_results.json`
- `tests/llm_toggle/venv_gpu_trace_rerun_after_flex/trace_run_results.json`

These are used for:
- OFF vs ON verdict deltas
- evidence retrieval trace inspection
- mini benchmark accuracy + neutral-rate comparison

## Scope Snapshot
- Path: `tests/README.md`.
- Purpose: Test suite scope and where to run validations.
- Audience: Engineers running, extending, or evaluating this module.

