# StreamingLLM Experiment Framework

Standalone framework to reproduce four experiments described in `experiment-design.md`.

## Structure
- `scripts/`: runnable entrypoints for data prep and experiments
- `streamingllm_experiment/`: reusable utilities (data, cache, eval, plotting)
- `configs/`: LLaMA-Factory pretraining configs for Experiment 4
- `data/`: datasets and generated corpora (repo-local by default)
- `outputs/`: logs, plots, and result tables

## Quick start
1. Install deps from `requirements.txt` in your environment.
2. Prepare data:
   - `python scripts/prepare_pg19.py`
   - `python scripts/generate_streameval.py`
3. Run experiments:
   - `python scripts/run_exp1_ppl.py`
   - `python scripts/run_exp2_rope_align.py`
   - `python scripts/run_exp3a_streameval.py`
   - `python scripts/run_exp3b_longbench.py`
4. Experiment 4:
   - Apply model patch and sink token utilities, then use configs under `configs/exp4/` with LLaMA-Factory.
   - Run `python scripts/run_exp4_eval_ppl.py` on trained checkpoints.

## Notes
- Llama-2 models are gated; ensure HF auth token is configured.
- Default paths are repo-local `data/` and `outputs/` but can be overridden by CLI args.
