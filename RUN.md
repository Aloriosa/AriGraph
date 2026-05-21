# RUN cookbook

All runnable commands for this repo. One source of truth — when a script changes or a new flag is added, update this file in the same diff.

Sections:
1. vLLM setup
2. From-scratch reproduction
3. Continual reproduction
4. PaperBench evaluation
5. Analysis
6. Utilities / TODO

---

## 1. vLLM setup

Launch the local vLLM server on GPUs 6/7 (Qwen3-Next-80B-A3B-FP8, port 1337).

```bash
./vllm.sh
```

Notes:
- Shebang inside the script must be `#!/bin/bash` (not `/bin/sh`) — uses bashisms.
- Do **not** pass `--enforce-eager` for real runs (kills TP=2 throughput).
- Keep `HF_TOKEN` in `.env`, not inlined into the script.

---

## 2. From-scratch reproduction

Run single-paper reproduction (from-scratch baseline) over a list of papers. Three phases run in order: triplet extraction → code→triplet linking → code generation. Outputs (graphs, `submission/`, log) land under `<log-path>/<paper>/`.

Resume behaviour (no extra flags): if `<log-path>/<paper>/paper_graph_data.json` and `mem_graph_data_with_code.json` already exist, extraction is **skipped** and the loaded graphs are reused; linking runs only when `triplet2code` is still empty; generation always runs. So re-running the same command resumes at linking → generation.

`--repo-dir` points at the **original authors' repo** (from-scratch links triplets to that repo). `--use-repo` is the alternative that clones the URL from the paper's `blacklist.txt`; don't pass both.

```bash
for pp in adaptive-pruning pinn fre mechanistic-understanding bridging-data-gaps \
          test-time-model-adaptation sample-specific-masks sapg stochastic-interpolants \
          bbox self-composing-policies stay-on-topic-with-classifier-free-guidance \
          rice lca-on-the-line all-in-one sequential-neural-score-estimation \
          robust-clip what-will-my-model-forget lbcs bam ftrl \
          self-expansion semantic-self-consistency; do
  python test_paper_reproduction.py --paper "$pp" --device cpu \
    --log-path rewritten-retrieval_generation-obs-by-codebase-qwen3-next-80b-ff8 \
    --repo-dir /home/asagirova/arigraph/cookbook_per_section_short_triplets_general_prompt_with_code/"$pp"/repo
done
```

Single paper (e.g. resume all-in-one at linking → generation after its graphs are already on disk):

```bash
python test_paper_reproduction.py --paper all-in-one --device cpu \
  --log-path rewritten-retrieval_generation-obs-by-codebase-qwen3-next-80b-ff8 \
  --repo-dir /home/asagirova/arigraph/cookbook_per_section_short_triplets_general_prompt_with_code/all-in-one/repo
```

Note: `--generate-code` is declared `type=bool`, so `--generate-code false` does **not** disable generation (`bool("false")` is `True`). Generation currently always runs. TODO: convert to an explicit `true/false` switch per the flag convention before relying on an off-switch.

Embedding cache (default ON): triplet, entity-label, and observation embeddings are cached to a `<json>_emb.npz` sidecar so reloading a paper/memory graph does **no** re-embedding on a warm hit. First load of an un-cached graph still embeds (and writes the sidecar); subsequent loads restore it. Old triplets-only sidecars are auto-upgraded to the full format on first load. Disable with `--emb-cache false`.

LLM per-call timeout (default 300s): each extraction/generation call to vLLM is bounded by `--llm-timeout-s`. Without it a stalled (queued on the shared GPUs) or repetition-looping call never raises, so the chunk-extraction retry loop can't fire and a single call blocks for many minutes. On timeout the call raises `APITimeoutError`, the retry loop catches it and retries. Loosen with `--llm-timeout-s 600` for legitimately long calls.

---

## 3. Continual reproduction

Continual run over an ordered paper list, with a bootstrap memory graph from a prior single-paper run. Outputs and cumulative graph checkpoints land under `continual_runs/my_run/<paper>/`.

```bash
python continual_paper_reproduction.py --resume \
  --bootstrap-mem-json /home/asagirova/arigraph/rewritten-retrieval_generation-obs-by-codebase-qwen3-next-80b-ff8/pinn/mem_graph_data_with_code.json \
  --papers pinn fre mechanistic-understanding stochastic-interpolants bbox rice lbcs bam ftrl \
           self-expansion semantic-self-consistency bridging-data-gaps sample-specific-masks \
           self-composing-policies adaptive-pruning lca-on-the-line robust-clip \
           stay-on-topic-with-classifier-free-guidance test-time-model-adaptation \
           what-will-my-model-forget \
  --paper-graph-base /home/asagirova/arigraph/rewritten-retrieval_generation-obs-by-codebase-qwen3-next-80b-ff8 \
  --output-root /home/asagirova/arigraph/continual_runs/my_run
```

Notes:
- `--resume` auto-detects the last completed paper and continues from there.
- Bootstrap paper choice is currently ad-hoc; see *Open design questions* in `CLAUDE.md`.
- Triplet-embedding cache is ON by default (reuses `<json>_emb.npz` sidecars across papers/resumes); disable with `--emb-cache false`.

---

## 4. PaperBench evaluation

Runs the PaperBench harness against a folder of generated code submissions. Per-run JSON score files land under `~/arigraph/frontier-evals/project/paperbench/runs/`.

Template (set the four variables, then run):

```bash
cd ~/frontier-evals/project/paperbench
source .env

BENCH_PAPER_SPLIT=all                  # name of .txt under experiments/splits
N_TRIES=1                              # eval runs per paper
SKIP_REPRO=false                       # true | false
CODE_ONLY=false                        # true | false
CODEGEN_RESULTS_FOLDER=~/kg4code/kg4code/rewritten-retrieval_generation-obs-by-codebase-qwen3-next-80b-ff8

uv run python -m paperbench.nano.entrypoint \
  paperbench.paper_split=${BENCH_PAPER_SPLIT} \
  paperbench.n_tries=${N_TRIES} \
  paperbench.solver=paperbench.solvers.direct_submission.solver:PBDirectSubmissionSolver \
  paperbench.solver.submissions_dir=${CODEGEN_RESULTS_FOLDER}/ \
  paperbench.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
  paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  paperbench.solver.computer_runtime.env.pull_from_registry=false \
  paperbench.reproduction.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
  paperbench.reproduction.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  paperbench.reproduction.computer_runtime.env.pull_from_registry=false \
  paperbench.reproduction.skip_reproduction=${SKIP_REPRO} \
  paperbench.judge.completer_config=preparedness_turn_completer.oai_completions_turn_completer:QwenOpenAICompletionsTurnCompleter.Config \
  paperbench.judge.grade_locally=true \
  paperbench.judge.scaffold=simple \
  paperbench.judge.code_only=${CODE_ONLY} \
  runner.recorder=nanoeval.json_recorder:json_recorder
```

---

## 5. Analysis

### Token budget comparison (continual vs from-scratch)

Parses log files and produces a per-paper CSV. Sign convention: `scratch − continual` (positive = continual saves tokens), ratio `scratch / continual` (>1 = continual cheaper).

```bash
python /home/asagirova/arigraph/compare_codegen_token_budgets.py \
  --continual-log /home/asagirova/arigraph/continual_runs_qwen3.6/log.txt \
  --scratch-root /home/asagirova/arigraph/rewritten-retrieval_generation-obs-by-codebase-qwen3-next-80b-ff8 \
  --out-csv /home/asagirova/arigraph/continual_runs_qwen3.6/token_budget_compare.csv
```

---

## 6. Utilities / TODO

- **Graph visualization** — `graphvis.ipynb`: TODO document the canonical cells to run.
- **PaperBench score extractor** — TODO: add a script that reads `frontier-evals/project/paperbench/runs/*.json` and appends one row per (paper, run) into the experiment CSV (see *Token budget accounting* in `CLAUDE.md`).
- **Push CSVs to Google Sheets** — TODO next session: generic `push_csv_to_gsheet.py {csv_path} {sheet_id} {tab_name}` that upserts by `paper` column. First user: token budget CSV from §5. Decide auth method (service account vs OAuth) before implementing.
- **Other ad-hoc analyses** — add commands here as they appear.
