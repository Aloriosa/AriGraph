# Paper Reproduction Research — Project Guide for Claude

## What this repo is

Research codebase for developing a **graph-based memory module for continual AI research replication**. The memory graph accumulates reusable code patterns, techniques, and best practices as it processes one ML paper after another; later papers benefit from this accumulated knowledge during code generation.

**Goals:**
- **Decrease code-generation token budget** for reproducing a paper.
- **Increase PaperBench quality** of generated code, by reusing best-practice code and techniques accumulated from continual implementation of prior papers.

**Regimes studied:**
- **From-scratch (baseline)** — code generation conditioned on a *single-paper* memory graph, no continual accumulation. Used to isolate the contribution of continual memory.
- **Continual** — a single theory graph accumulates across many papers. Triplets extracted from generated code for paper *N* feed back into the graph used for paper *N+1*.

Evaluation target: **PaperBench**.

## Status & how to engage critically

This is an **active work-in-progress research project**. The idea, motivation, and architecture are all still being shaped — the user explicitly wants persistent revising, critique, and new ideas to reduce flaws and make the system more intuitive, reliable, and well-motivated.

**What this means for our collaboration:**
- **Be a research collaborator, not an order-taker.** Push back on weak motivation, ambiguous problem framings, and architectural choices that don't pull their weight.
- **Surface flaws actively.** If a method step looks unmotivated, brittle, or redundant, say so. If an experiment wouldn't actually support the claim being made, say so.
- **Propose alternatives.** When critiquing, also offer 1–2 concrete alternative framings, designs, or experiments.
- **Suggest new ideas.** Look across the repo and conversation for opportunities — ablations, baselines, analyses, simpler architectures, sharper metrics, related literature worth comparing against.
- **Revise relentlessly.** Treat existing code and design docs as drafts. Suggest renames, restructures, and deletions when they would make the project more intuitive. Don't preserve cruft out of politeness.
- **No sycophancy.** Don't agree to keep the user happy. If a direction looks wrong, say so plainly and explain why.

## Active drivers

- **`continual_paper_reproduction.py`** — main continual driver. Loads a bootstrap theory graph from one paper, then for each subsequent paper: runs topology-aware retrieval + code generation, extracts new triplets from generated code, links code to triplets, saves the cumulative memory graph for resumability. The bootstrap paper (passed via `--bootstrap-mem-json`) is currently chosen ad-hoc / randomly. This is an experimental variable that needs ablation, not a fixed convention — see Open design questions.
- **`test_paper_reproduction.py`** — single-paper (from-scratch baseline) pipeline; also reused as a library by the continual driver.
- **`topology_pipeline/`** — modular components shared by both: `retrieval.py`, `prompt_packing.py`, `schemas.py`, `symbol_assets.py`, `graph_materialization.py`, `cards.py`.
- **`continual_learning_graph.py`** + `CONTINUAL_LEARNING_README.md` — alternative continual-learning graph design (entity resolution + aggregation + pattern mining). Status: TODO — confirm whether active or superseded.

**Read `continual_paper_reproduction.py` first** to orient.

## Resumability convention (continual runs)

Continual reproduction runs can span many papers and many hours; crashes mid-run must not lose accumulated graph state.

- **Memory graph checkpoints:** after processing each paper, the cumulative memory graph is saved under `{output_root}/{paper_slug}/mem_graph_data_with_code.json` (same schema as the single-paper output). A run can be resumed by pointing `--bootstrap-mem-json` at the last successfully-saved checkpoint and trimming `--papers` to the remaining list.
- **Per-paper outputs:** all retrieval/generation artefacts for paper *N* live under `{output_root}/{paper_slug}/` — safe to inspect or delete to force re-processing of a single paper.
- **Idempotence:** the continual update only adds triplets/links that don't already exist, so re-running over an already-processed paper is safe but wasteful; prefer resuming via the `--resume` flag.
- **Resuming:** `continual_paper_reproduction.py --resume ...` auto-detects the last completed paper and continues from there. Canonical command lives in `RUN.md`.

## Repo layout (key dirs)

- `agents/` — LLM client wrappers (`GPTagent`, `llama_agent`).
- `graphs/` — base graph class and subclasses (`parent_graph.py`, `contriever_graph.py`).
- `topology_pipeline/` — modular retrieval / packing / schema / symbol-indexing.
- `prompts/cookbook_extraction_prompt.py`, `prompts/paper_reproduction_prompt.py` — extraction + generation prompts.
- `utils/`, `utils_reproduction.py` — service code; `Logger` (in `utils/utils.py`) is the standard logger.
- `frontier-evals/project/paperbench/` — PaperBench papers and eval harness.
- `repos/`, `repos-for-grading/` — target code repositories per paper.
- `continual_runs/`, `continual_runs_qwen3.6/`, `runs_from_scratch/`, `runs_from_scratch-wen36/`, `runs_fr*/` — experimental run outputs.
- `submission_emnlp/` — paper draft (LaTeX). **Findings flow into here during experiments, not after.**

## Stack

- Python 3.11+, PyTorch, HuggingFace.
- **vLLM** for local inference (Qwen3-Next-80B-A3B-FP8). OpenAI-compatible server on port 1337.
- LaTeX for the paper draft.
- No package — everything runs as scripts from the repo root.

### Conda env — CRUCIAL

**Always run code/tests from a bash terminal with the `arigraph` conda env active.** The base/default `python` lacks the project deps (`openai`, torch, vLLM client, etc.) — running without the env will fail with `ModuleNotFoundError`. Activate first, every time:

```bash
conda activate arigraph
```

Since shell state does not persist between separate command invocations, run a single combined command when activation is needed in a fresh shell:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate arigraph && python <script>.py ...
```

Never verify a script "works" with the default interpreter — it gives a false negative.

## Hardware notes

- User uses **GPUs 6/7** for vLLM, NV12 NVLink between them.
- Other GPUs on the node may be used by others; check `nvidia-smi` before launching.
- Scripts talk to vLLM at `http://localhost:1337`.

### vLLM launch gotchas

- Shebang must be `#!/bin/bash`, not `#!/bin/sh` — uses bashisms (`set -o pipefail`, `{1..10}`).
- `--enforce-eager` kills TP=2 throughput — only use for debugging.
- `NCCL_P2P_LEVEL=SYS` and `VLLM_SKIP_P2P_CHECK=1` are debug workarounds, not defaults — with NV12 they should not be needed.
- Set `CUDA_VISIBLE_DEVICES=6,7` in the launch env if not relying on shell defaults.

## Secrets

- Secrets (HF tokens, API keys, SSH keys) live in **`.env*` files only**, never inlined into shell scripts, Python, or any committed file.
- `.env*` files are gitignored — verify before committing.
- **Existing leak risk in repo (TODO to clean up):**
  - `vllm.sh` has `HUGGING_FACE_HUB_TOKEN=hf_…` hardcoded — move to `.env`.
  - `mlspace__private_key.txt` is sitting at repo root — confirm it's gitignored and move out of repo if possible.
- If you (Claude) see a hardcoded token, key, or credential in any file, **stop and flag it immediately** — do not edit around it as if it were normal.

## PaperBench evaluation

Eval scripts live spread across `graphvis.ipynb` and `~/frontier-evals/project/paperbench/`. The canonical command template (used to generate concrete eval commands) is:

```bash
cd ~/frontier-evals/project/paperbench
source .env
uv run python -m paperbench.nano.entrypoint \
  paperbench.paper_split={bench_paper_split} \
  paperbench.n_tries={n_tries} \
  paperbench.solver=paperbench.solvers.direct_submission.solver:PBDirectSubmissionSolver \
  paperbench.solver.submissions_dir={codegen_results_folder}/ \
  paperbench.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
  paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  paperbench.solver.computer_runtime.env.pull_from_registry=false \
  paperbench.reproduction.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
  paperbench.reproduction.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  paperbench.reproduction.computer_runtime.env.pull_from_registry=false \
  paperbench.reproduction.skip_reproduction={skip_repro} \
  paperbench.judge.completer_config=preparedness_turn_completer.oai_completions_turn_completer:QwenOpenAICompletionsTurnCompleter.Config \
  paperbench.judge.grade_locally=true \
  paperbench.judge.scaffold=simple \
  paperbench.judge.code_only={code_only} \
  runner.recorder=nanoeval.json_recorder:json_recorder
```

**Key knobs:**
- `paper_split` — name of a `.txt` split file under `frontier-evals/project/paperbench/experiments/splits/` (e.g. `all`).
- `n_tries` — number of code-result runs to evaluate per paper.
- `submissions_dir` — folder of generated code to be graded (one of our `retrieval_generation-...` / `rewritten-retrieval_generation-...` dirs).
- `skip_reproduction` (`true`/`false`) — whether to skip actually re-running the generated code.
- `code_only` (`true`/`false`) — grade code-quality only vs. full reproduction.
- Judge: local grading with a Qwen completer; recorder writes JSON.

**Output location:** each eval run writes a JSON score file under `frontier-evals/project/paperbench/runs/`. One run per invocation; paper-level scores are inside the JSON. **TODO:** confirm the JSON schema (key names for paper id, score, n_tries breakdown) so we can write a tiny extractor that appends one row per (paper, run) into the experiment CSV.

## Token budget accounting

**Goal:** quantify whether continual memory reduces *code-generation* token cost vs the from-scratch baseline, at matched-or-better PaperBench quality.

**What is counted (per phase, prompt and completion separately):**
- `extraction` — LLM calls extracting triplets from paper / code.
- `retrieval` — any LLM calls during retrieval (if applicable; embeddings tracked separately).
- `generation` — code-generation calls (the headline phase the project aims to shrink).
- `linking` — code↔triplet linking calls.
- Plus per-phase and grand totals.

Embedding tokens (cheap, not LLM-billed in the same sense) live in an optional separate column — never summed into the LLM totals.

**Granularity:** **per-paper is primary** (one paper = one PaperBench sample). Aggregation across papers happens downstream from the CSV — never bake aggregation into the logging.

**Storage:**
- Token counts are currently emitted into per-run `log.txt`. `compare_codegen_token_budgets.py` parses logs into a CSV.
- Long term: emit token counts into a structured per-paper JSON or directly into the CSV at runtime, so log-parsing isn't load-bearing. (TODO.)

**Headline metric:** `gen_tokens(from_scratch) − gen_tokens(continual)`, per-paper. **Positive = continual saves tokens** (the current script has the wrong sign — fix needed in `compare_codegen_token_budgets.py`).

**Caveats to communicate every time we report this:**
- Continual amortizes earlier extraction/linking costs across papers. Report both marginal-per-paper cost and cumulative whole-run cost.
- Continual may grow the *prompt* even as the *completion* shrinks — always report both.
- Token savings are only meaningful at matched-or-better PaperBench score. Default reporting view: **Pareto frontier of (tokens, score)**, continual vs from-scratch, not single-axis token savings.

**Proposed CSV schema** (one row per `(paper_id, regime, bootstrap_id, seed, run_id)`):
```
extraction_prompt, extraction_completion,
retrieval_prompt,  retrieval_completion,
generation_prompt, generation_completion,
linking_prompt,    linking_completion,
total_prompt, total_completion, total_all,
embedding_tokens (optional),
paperbench_score, paperbench_n_tries,
wallclock_s
```
Storing everything raw means any later aggregation is a pandas one-liner — no rerunning experiments.

## How to work with me here

### Workflow
1. **Brainstorm** before coding any new experiment (clarify intent, scope, success criteria).
2. **Write a plan** before touching code on multi-step tasks.
3. **Ask** when unsure — do not guess.
4. **Verify** by running and showing real output before claiming work is done.
5. **Small, reviewable diffs.**

### Code style
- **Flat readable Python** — code should tell the story of the method.
- **No try/except** unless explicitly requested — let it crash.
- **No type hints.**
- **Short functions**; classes only when genuinely needed.
- **Logging: stdout + file simultaneously** (every script). Use `Logger` from `utils/utils.py`.
- **Comments: minimalistic but comprehensive.** Explain WHAT and WHY, but in the fewest words that still carry the meaning. One short line above a non-trivial block is better than a paragraph; one inline note next to a design decision is better than restating the variable name. Cover every step that isn't self-evident — but say it tersely. No redundant restatements of obvious code, no decorative banners, no multi-paragraph docstrings.
- **Flag naming: always explicit two-way switches.** Every boolean flag takes a value: `--foo true` / `--foo false`. **Never** use one-way `store_true` / `store_false` flags or `--no-foo` patterns — they're asymmetric, easy to misread, and the value isn't visible in the command. The same holds for value-bearing flags: always pass the value explicitly. Default must be set in the argparse spec (`default=True` or `default=False`), and the *default must be ON* for newly added features.
- **New features: flag exists, defaults ON, off-switch surfaced.** Every new feature gets a CLI flag (or config knob) so it can be toggled — never bake it in unconditionally. The flag's **default is ON** so the user gets the new behavior without remembering an incantation. When you add the flag, **explicitly tell the user the exact command to disable it** (e.g., `--foo false`) alongside the normal run command. Update `RUN.md` in the same diff to show both the default invocation and the off-switch form.
- **Surface new flags / commands explicitly.** Whenever a new CLI flag, env var, config knob, or invocation pattern is introduced, **tell the user the exact command to run** (default-on form *and* off-switch form), including its place in the existing workflow. Never assume the user will rediscover an option by reading `--help`.
- **No silent hardcoding.** Do not bake values into code unless the user explicitly says so. Every tunable belongs as a function argument, CLI flag, or config field with a clear name and a sensible default. Hidden magic numbers compound as the codebase grows and make it nearly impossible to debug why "the whole thing works completely other than intended." If a value is genuinely a constant, name it (UPPER_SNAKE_CASE at module top) and comment WHY that value.

### Single project-wide RUN cookbook
Maintain **one** file (`RUN.md` at repo root) containing all runnable commands for every script in the repo, stacked with minimalistic explanations. Format: a short heading per script, one or two lines saying what it does, then a fenced copy-pasteable command block. Group sections by regime: **vLLM setup**, **from-scratch reproduction**, **continual reproduction**, **PaperBench evaluation**, **analysis**, **utilities**. When a new script is added or a command changes, **update `RUN.md` in the same diff** — no separate per-script READMEs.

### Every experiment ends with
1. **CSV is the default artifact.** One row per run/condition, plot-ready columns — no nested structures. Key axes for this project: **generation token budget**, **PaperBench score**, regime (from-scratch vs continual), paper id, seed. The CSV alone is enough to "end" most small experiments.
2. **Stats: proportional, not performative.** Don't pile on means/stds/CIs/p-values for every small experiment — most of them are meaningless without enough seeds or large enough effects to matter. For a small exploratory run, just report the raw numbers from the CSV. Heavier statistical treatment (significance tests, CIs, multiple-seed aggregation) is a per-experiment design decision to brainstorm when it actually adds information. Always flag single-seed numbers as such.
3. **Plots only on request.** Do not auto-generate plots. The user will ask for plots when needed; until then, the CSV is the deliverable.
4. **Paper LaTeX update** in `submission_emnlp/` *in the same session* — new findings, tables (from CSV), and half-formed observations as TODO notes inside the draft.

When proposing an experiment, also propose: the CSV schema and the paper section the result will land in. Mention heavier stats or plots only if they're justified for that specific experiment.

### Git: commit after meaningful changes, with explicit approval
After each meaningful change, **prepare a commit and ask the user for approval before running `git commit`**. Don't auto-commit silently, and don't wait for the user to ask either — surface it.

"Meaningful" = a completed unit of work, not every keystroke:
- **Code:** after finishing a coherent change (a new script, a working refactor, a verified bug fix). Not after every Edit during iteration.
- **LaTeX:** after each substantive paper-draft update (new finding written up, new table/figure inserted, a section reorganized). Not for typo fixes mid-sentence.
- **Experiment outputs (CSVs, logs):** commit when they represent a finished experiment run.

**Approval protocol — every time, no exceptions:**
1. Run `git status` + `git diff --stat` (and full `git diff` for small changes) so the user can see exactly what's in scope.
2. Present a concise list: files to be staged, one-line summary of each change, proposed commit message.
3. **Wait for explicit user approval** (e.g. "yes", "go", "commit"). Silence or unrelated replies are not approval.
4. Only then `git add` the listed files and `git commit`.

Commit messages: short, descriptive, present tense. Reference the experiment / script / paper section. Group related edits (code + its `RUN.md` entry + its LaTeX mention) into one commit when they belong together.

**Never** commit secrets (`.env*`, `*private_key*`, HF tokens). **Never** force-push or amend without asking. **Never** push to remote unless the user asks.

## Things that may look weird but are intentional

- `from __future__ import annotations` with no type hints — intentional, do not add hints.
- Many CSVs at repo root are experiment outputs, not stale junk.
- TODO: user to confirm any other intentional choices that look like cleanup opportunities.

## Open design questions (revisit every session)

Currently unresolved — push back when work touches these, and propose answers.

- **What counts as a "reusable best practice" worth storing in the graph?** Is it any extracted triplet, or only ones that survived some quality filter? Should we weight by paper success? By frequency? By recency?
- **How do we measure that continual memory is actually helping?** Comparing PaperBench score vs from-scratch is necessary but not sufficient — graph size grows monotonically, so improvements could just be "more context." Need an ablation that isolates the *useful* portion.
- **Is triplet extraction from generated code the right signal?** Generated code may itself be wrong. Should extraction be gated on PaperBench-passing code only? Successful test execution? Some other quality proxy?
- **Token budget — input vs output, total vs per-call?** Defined above; left here as a flag that the *operational* definition (which dimensions get reported as "the" headline metric) is still in motion.
- **Bootstrap paper choice — needs ablation.** Currently picked randomly. We don't know how sensitive the final continual graph (and downstream PaperBench scores) is to the bootstrap. Concrete experiment: same `--papers` list, vary the bootstrap across N choices, measure variance in downstream scores. Until done, *any* continual-vs-from-scratch claim is confounded by this choice.
- **Catastrophic interference:** if late papers contradict early ones, does the graph degrade earlier-paper reproduction? Currently untested.

Add new questions here as they appear. Remove (don't just check off) ones that are conclusively resolved.

---

*Loaded automatically every session. Update when project shape changes meaningfully.*
