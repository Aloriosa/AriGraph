"""
Specialized prompts for extracting cookbook knowledge from research papers,
including implementation details, common errors, best practices, and gotchas.
Supports incremental knowledge graph construction across multiple papers.
"""

# =============================================================================
# MAIN COOKBOOK EXTRACTION PROMPT
# =============================================================================
"""
KNOWLEDGE CATEGORIES TO EXTRACT:

1. IMPLEMENTATION COMPONENTS:
   Extract models, architectures, modules, algorithms, and their relationships.
   Examples: 'Transformer, contains, encoder'; 'encoder, has layers, 6'; 'attention, type, multi-head'; 'module X, extends, module Y'; 'function X, computes, metric Y'.

2. HYPERPARAMETERS & CONFIGURATIONS:
   Extract ALL numerical settings, thresholds, parameters with their exact values.
   Examples: 'learning rate, value, 0.001'; 'batch size, value, 32'; 'dropout, rate, 0.1'; 'warmup steps, value, 4000'; 'parameter X, range, 0.1 to 0.5'.

3. EXPERIMENTAL SETUP:
   Extract datasets, hardware requirements, preprocessing steps, evaluation metrics.
   Examples: 'model, trained on, WMT2014'; 'training, requires, 8 V100 GPUs'; 'preprocessing, includes, tokenization'; 'evaluation, uses metric, BLEU'.

4. TRAINING PIPELINE:
   Extract sequential procedures, workflow steps, and their dependencies.
   Examples: 'training, step 1, data loading'; 'step 2, follows, step 1'; 'checkpoint, saved every, 1000 steps'; 'early stopping, after epochs, 10'.

5. DEPENDENCIES & REQUIREMENTS:
   Extract libraries, frameworks, versions, and prerequisites.
   Examples: 'implementation, requires, PyTorch 1.9'; 'model, depends on, CUDA 11'; 'baseline, includes, BERT'.

6. COMMON ERRORS & SOLUTIONS:
   Extract problems that can occur, their symptoms, causes, and fixes.
   Examples: 'NaN loss, caused by, exploding gradients'; 'NaN loss, solved by, gradient clipping'; 'OOM error, solved by, reduce batch size'; 'shape mismatch, occurs in, attention layer'.

7. BEST PRACTICES:
   Extract proven techniques and recommendations.
   Examples: 'gradient clipping, prevents, NaN loss'; 'warmup, improves, training stability'; 'layer norm, placed before, attention'; 'mixed precision, speeds up, training'.

8. GOTCHAS & PITFALLS:
   Extract non-obvious issues and warnings.
   Examples: 'attention mask, must be, additive not multiplicative'; 'positional encoding, must be, registered as buffer'; 'default init, causes, slow convergence'.

9. PERFORMANCE RESULTS:
   Extract metrics, scores, and comparisons.
   Examples: 'model, achieves BLEU, 28.4'; 'method, outperforms baseline, by 2.1 points'; 'training time, reduced by, 30%'.

"""



prompt_cookbook_extraction = '''Objective: The main goal is to meticulously gather implementation knowledge from research paper text and code to organize this data into a clear, structured knowledge graph that enables reproducing the research and avoiding common pitfalls.

Guidelines for Building the Knowledge Graph:

Creating Nodes and Triplets: Nodes should depict entities or concepts related to implementation. Use a structured triplet format to capture data, as follows: "subject, relation, object". For example, from "We use Adam optimizer with learning rate 0.001 and batch size 32, trained on ImageNet dataset," extract "optimizer, type, Adam; learning rate, value, 0.001; batch size, value, 32; model, trained on, ImageNet."
Remember that you should break complex triplets like "Transformer, uses attention with 8 heads and 512 dimensions" into simple triplets like "Transformer, uses, attention", "attention, num heads, 8", "attention, dimension, 512".
Length of your triplet should not be more than 7 words. You should extract only concrete knowledge, any assumptions must be described as hypothesis.
For example, from phrase "This approach might improve convergence speed" you should extract "approach, might improve, convergence speed" and should not extract "approach, improves, convergence speed".
Remember that object and subject must be atomic units while relation can be more complex and long.


Do not miss important information. If observation describes an error and its solution, triplets should include both: 'error X, has symptom, Y', 'error X, solved by, Z'. If observation mentions a best practice, include what it prevents: 'practice X, prevents, problem Y'.
There could be connections between distinct parts of observations. For example if there is information about a hyperparameter at the beginning and its effect on training at the end, you should extract triplets connecting them.
Several triplets can be extracted that contain information about the same node. For example 'Adam, is type, optimizer', 'Adam, learning rate, 0.001', 'Adam, beta1, 0.9'. Do not miss this type of connections.

Do not include triplets about general knowledge unrelated to implementation.
Any experiments that are only introduced in the Appendix are not considered core contributions of the paper, and so are out of scope. This is the case even if the experiment is referenced in the main text. HOWEVER, if an experiment is described in the main body of the text, but some of the details used (e.g. details about a prompt used, or a long mathematical expression) have been put in the appendix, then that experiment is considered still in scope. The reason here is that the experiment is described in the main body of the paper.

Do not use 'none' as one of the entities.
ALWAYS include units when mentioned (%, seconds, tokens, dimensions, GB, etc.).

Example of triplets you have extracted before: {example}

Input to process: {observation}

Remember that triplets must be extracted in format: "subject_1, relation_1, object_1; subject_2, relation_2, object_2; ..."

Extracted triplets: '''


prompt_cookbook_extraction_wo_code = '''Objective: Build a minimal, implementation-complete knowledge graph for yourself that will later reproduce the paper's method and all the experiments and results using ONLY this graph as source of truth (without access to the paper).

{observation}

Extraction Priority (highest to lowest):
1) Exact procedure/algorithm steps and ordering constraints.
2) Hyperparameters, schedules, thresholds, stopping criteria, and optimization settings.
3) Data setup: sources, filtering, preprocessing/tokenization, and train/val/test splits.
4) Architecture/module composition and dependencies between components.
5) Evaluation protocol: metrics, baselines, inference/decoding settings.
6) Failure modes, diagnostics, ablations tied to implementation choices, and fixes.

Canonicalization and Token-Efficiency Rules:
- Keep subject/object atomic, normalized, and consistent across chunks.
- Prefer canonical names (e.g., learning rate -> learning_rate) to avoid duplicate entities.
- Reuse existing entities from prior triplets whenever possible.
- Use canonical relation families whenever possible:
  - structure: part_of, has_component
  - dependency: uses, requires, depends_on
  - configuration: configured_by, has_config, value
  - procedure: precedes, executes
  - data/eval: trained_on, evaluated_by, achieves
  - diagnostics: fails_on, caused_by, fixed_by, solved_by
  - mapping: implemented_in, implemented_by, defined_in
- Favor typed entities to make graph topology useful. Suggested entity classes:
  module, algorithm_step, hyperparameter, schedule, metric, dataset, artifact, script, failure_mode, fix, symbol.
- If uncertain, encode uncertainty explicitly in relation (e.g., might improve, hypothesis).
- Avoid generic background knowledge not needed for implementation.
- Skip facts already present in previous triplets unless adding more specific detail.

Guidelines for Building the Knowledge Graph:

Creating Nodes and Triplets: Nodes should depict entities or concepts related to implementation. Use a structured triplet format to capture data, as follows: "subject, relation, object". For example, from "We use Adam optimizer with learning rate 0.001 and batch size 32, trained on ImageNet dataset," extract "optimizer, type, Adam; learning rate, value, 0.001; batch size, value, 32; model, trained on, ImageNet."
Remember that you should break complex triplets like "Transformer, uses attention with 8 heads and 512 dimensions" into simple triplets like "Transformer, uses, attention", "attention, num heads, 8", "attention, dimension, 512".
Length of your triplet should not be more than 7 words. You should extract only concrete knowledge, any assumptions must be described as hypothesis.
For example, from phrase "This approach might improve convergence speed" you should extract "approach, might improve, convergence speed" and should not extract "approach, improves, convergence speed".
Remember that object and subject must be atomic units while relation can be more complex and long.


Do not miss important information. If observation describes an error and its solution, triplets should include both: 'error X, has symptom, Y', 'error X, solved by, Z'. If observation mentions a best practice, include what it prevents: 'practice X, prevents, problem Y'.
There could be connections between distinct parts of observations. For example if there is information about a hyperparameter at the beginning and its effect on training at the end, you should extract triplets connecting them.
Several triplets can be extracted that contain information about the same node. For example 'Adam, is type, optimizer', 'Adam, learning rate, 0.001', 'Adam, beta1, 0.9'. Do not miss this type of connections.

Do not include triplets about general knowledge unrelated to implementation.
Any experiments that are only introduced in the Appendix are not considered core contributions of the paper, and so are out of scope. This is the case even if the experiment is referenced in the main text. HOWEVER, if an experiment is described in the main body of the text, but some of the details used (e.g. details about a prompt used, or a long mathematical expression) have been put in the appendix, then that experiment is considered still in scope. The reason here is that the experiment is described in the main body of the paper.

Do not use 'none' as one of the entities.
ALWAYS include units when mentioned (%, seconds, tokens, dimensions, GB, etc.).
Target output budget efficiency where possible; prioritize high-value implementation facts first.

Example of triplets you have extracted before: {example}

Remember that triplets must be extracted in format: "subject_1, relation_1, object_1; subject_2, relation_2, object_2; ..."

Extracted triplets: '''


# =============================================================================
# PROMPT FOR PAPER-TO-CODE LINKING (Phase 2: connect paper concepts to code)
# =============================================================================

prompt_paper_code_linking = '''Objective: Given a paper knowledge graph and a codebase, create triplets that LINK paper concepts to their implementations in code.

Paper graph (implementation concepts from the paper):
{paper_graph_triplets}

Code index for this file:
{code_index}

Code for file {file_path}:
{code_chunk}

Rules:
- Create triplets: (paper_entity, relation, code_entity)
- code_entity format: "path/to/file" or "path/to/file::ClassName" or "path/to/file::function_name"
- Relations: implemented_in, configured_in, handles, executes, defined_in
- Only link when the code clearly implements/handles the paper concept
- Reuse existing paper entity names from the graph exactly
- Do not invent new paper entities
- If no clear links exist for this file, return empty (no triplets)

Output format: "subject, relation, object; subject, relation, object; ..."

Linking triplets: '''


# =============================================================================
# PHASE 1: REUSABLE COOKBOOK EXTRACTION (extract only reusable patterns from paper)
# =============================================================================

prompt_cookbook_reusable_extraction = '''You are building a REUSABLE COOKBOOK for a research area. This cookbook will help a coding agent implement papers in this area.

You observe ONE paper. Extract ONLY what is REUSABLE across similar papers — patterns, pipelines, building blocks that would help implement OTHER papers. Skip paper-specific details.

EXCLUDE: Motivation, related work, citations. Also EXCLUDE: one-off hyperparameters, paper-specific experiment names (e.g. "Table 2"), exact result numbers.

INCLUDE (only if reusable):
1) COMMON PIPELINE: Standard steps (e.g. load_data, preprocess, train, eval) that appear in many papers.
2) REUSABLE ALGORITHMS: Patterns and building blocks (e.g. Transformer blocks, attention, loss types) — use generic names.
3) TYPICAL CONFIG: Common hyperparameters and their typical ranges/values (e.g. learning_rate, batch_size, warmup_steps).
4) EVALUATION PROTOCOL: Standard metrics and setups (e.g. perplexity, BLEU, train/val split).
5) CODE-ALIGNED CONCEPTS: entities that can be linked to symbols/classes/functions in a typical repo.

Ask: "Would an implementer of a DIFFERENT paper in this area benefit from this?" If no, skip.

Format as triplets: "subject, relation, object".
Use generic/canonical names, not paper-specific ones.
Use topology-friendly relations: part_of, uses, requires, configured_by, has_config, precedes, trained_on, evaluated_by, implemented_by.
Prefer canonical names for entity aliases: learning_rate, batch_size, val_split, top_k, mixture_of_experts.

Example of triplets you have extracted before: {example}

Input to process: {observation}

Remember that triplets must be extracted in format: "subject_1, relation_1, object_1; subject_2, relation_2, object_2; ..."

Extracted triplets: '''


# =============================================================================
# PHASE 2: REUSABLE PATTERN TO CODE MAPPING (hypernodes)
# =============================================================================

prompt_reusable_pattern_to_code_mapping = '''You have excerpt from the paper and REUSABLE PATTERNS for reprodution extracted from the excerpt.
You have the paper's codebase.

Excerpt from the paper:
{observation}

Reusable patterns from the cookbook:
{reusable_patterns}

{codebase}

Your task: For each reusable pattern, produce a HYPERNODE — a structured entry containing:
1) code: the exact implementation (class or function body, or config snippet)
2) documentation: brief description (1-3 sentences) of what it does and when to use it
3) imports: list of required packages/imports (e.g. ["torch", "transformers", "from datasets import load_dataset"])

Think: "Where would a developer implementing ANY paper in this area find the implementation of pattern X?"
For each match, extract the actual code, write brief docs, and list imports.


Output format: For each (pattern, hypernode), output a JSON block:
{{"pattern": "...", "hypernode": {{"code": "...", "documentation": "...", "imports": [...]}}}}

Only output for REUSABLE patterns. Skip paper-specific implementations.
If this file does not implement any reusable pattern, output nothing.
Output one JSON block per pattern, or nothing if no matches.'''


# =============================================================================
# PHASE 2 (alt): SYMBOL INDEX + STRUCTURED LINKS (triplet_id → file:line)
# =============================================================================

prompt_code_linking_symbol_index = '''You map numbered knowledge-graph triplets to concrete code locations in a repository.

Paper excerpt (context):
{observation}

Numbered triplets (reference by index only — use field triplet_id matching the number before each parenthesis):
{numbered_triplets}

Repository symbol index (files, classes, methods; line numbers are 1-based):
{symbol_index}

Task: For each triplet that is implemented in this codebase, output ONE link object with:
- triplet_id: integer — must match a number from the numbered list above
- code_location: string — MUST be "relative/path/from/repo.py:LINE" where LINE is the start line of the defining symbol (class, method, or function) that best implements the triplet
- confidence: float between 0 and 1

Rules:
- Use only paths that appear in the symbol index. Prefer the smallest enclosing symbol (method vs whole class) when appropriate.
- Do not invent file paths or line numbers.
- If a triplet has no clear implementation in the repo, omit it.
- Output a single JSON object with key "links" whose value is an array of objects. Example:
{{"links": [{{"triplet_id": 1, "code_location": "src/model.py:42", "confidence": 0.85}}]}}
- If nothing matches, return {{"links": []}}.'''


# =============================================================================
# CODING AGENT PROMPT (uses paper graph + expert graph, benchmark-compliant)
# =============================================================================

BENCHMARK_EVALUATION_RULES = '''The output of your work must be a git repository containing all the necessary components to reproduce the results of the paper. Your repository should be located at `/home/submission/`, and the committed files should not exceed 1GB in size (i.e. you should only commit source code without any heavy artifacts). We will run `git clean -fd` on the submission directory before grading it, so any untracked files will be lost.

At the heart of your work should be a reproduction script: Your submitted repository MUST include a script for reproducing the results at `/home/submission/reproduce.sh`. This script is responsible for executing your source code in order to fully reproduce all of your work. We will copy your submission to a fresh Ubuntu 24.04 LTS Docker container and run `bash reproduce.sh` from the submission directory, for a maximum runtime of 7 days. Your submission may not be placed at the same path where you submitted it, so do not rely on hardcoded absolute paths to files in your codebase. The container will have access to an NVIDIA A10 GPU, with the NVIDIA container toolkit already installed. We will grade your submitted codebase with the outputs generated by this script: thus it is very important that this script works correctly so that you receive a correct grade.

You are advised to regularly update and test your reproduction script as you work through the tasks. Docker has been installed in your environment, should you wish to use it.

Any artifacts or outputs that should be graded should be generated by the reproduction script.

Finally, please also include a README.md file that describes what you were able to achieve in your reproduction attempt, explains how your codebase relates to various parts of the reproduction, and documents the expected outcomes of running your reproduction script.'''

REPRODUCTION_SCRIPT_TOY_EXAMPLE = '''
REPRODUCTION SCRIPT TOY EXAMPLE
------

**Imagine the following toy paper**:

```
"We count the number of 'r's in the word 'strawberry' using a python script, and find that there are 3 instances of the letter 'r'"
```

> the paper measures the number of 'r's in the word strawberry (this is an artefact, think of this as a table or figure or result), using a basic python script as an implementation (think of this as an algorithm described in a paper)

**Toy Submission**:

FILE: submission/count.py

```python
import argparse, csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word', default="strawberry")
    parser.add_argument('--output', default="output.csv")
    args = parser.parse_args()

    r_count = args.word.lower().count('r')
    with open(args.output, 'w', newline='') as f:
        csv.writer(f).writerows([["word", "r count"], [args.word, r_count]])

    print(f"'{args.word}' has {r_count} 'r'(s). Saved to '{args.output}'.")

if __name__ == "__main__":
    main()
```

FILE: submission/reproduce.sh

```bash
apt-get update && apt-get install -y python3

# Run the Python script with the specified arguments
python3 count.py --word strawberry --output output.csv

# Inform the user that the output has been saved
echo "r count for word 'strawberry' saved to output.csv"
```

**How we would grade this**:
- Implementation: correct, a python script for counting "r"s in "strawberry" was implemented
- Execution: correct, the python script for counting "r"s in "strawberry" was correctly **executed by the reproduce.sh** script
- Result: inspecting the output.csv **produced by reproduce.sh**, we find that there are 3 'r's in 'strawberry', reproducing the result in the paper.
'''

prompt_coding_agent_paper_mem_graph = '''You are a coding agent implementing a research paper. You have access to:

1) **PAPER TRIPLETS** — triplets extracted from the target paper containing implementation details (algorithms, hyperparameters, setup, procedures).
2) **EXPERT KNOWLEDGE** — reusable reproduction pattern triplets and code snippets from similar papers related to given paper triplets (pipelines, code, documentation, imports). Use these as drop-in templates.

Your strategy:
1) USE EXPERT KNOWLEDGE as a starting point for your implementation. Use patterns as a guidance for the code applicability areas. Use the code snippets as building block templates for your implementation.
2) Use PAPER TRIPLETS to understand the paper's context, method details and requirements.
3) FOCUS on what the PAPER TRIPETS specify: the method, architecture, hyperparameters, and procedure unique to this paper.
4) COMBINE: wire the paper-specific parts into the expert knowledge code.

{BENCHMARK_RULES}

{TOY_EXAMPLE}

---
{expert_knowledge}

The expert knowledge is structured in three blocks:
1) paper implementation cards (requirements),
2) retrieved reusable memory cards (patterns),
3) linked code assets (selected snippets).
Respect this order: satisfy paper requirements first, then adapt reusable patterns, then use code assets.

---
Generate a complete, runnable repository. Include reproduce.sh, README.md, and all source code. Use expert graph patterns where applicable. Implement paper-specific parts from the paper graph. Output format:
FILE: path/to/file.ext
```language
<content>
```
'''

prompt_coding_agent_paper_graph = '''You are a coding agent implementing a research paper. You have access to ONE knowledge graph:

**PAPER GRAPH** — implementation details extracted from the target paper (algorithms, hyperparameters, setup, procedures).

Your strategy:
USE THE PAPER GRAPH TO IMPLEMENT THE PAPER: data loading, training loop, config, evaluation, the method, architecture, hyperparameters, and procedure unique to this paper.

{BENCHMARK_RULES}

{TOY_EXAMPLE}

---
PAPER TO IMPLEMENT (summary):
{paper_summary}

---
PAPER GRAPH (implementation details from the paper):
{paper_graph}

---
Generate a complete, runnable repository. Include reproduce.sh, README.md, and all source code. Use expert graph patterns where applicable. Implement paper-specific parts from the paper graph. Output format:
FILE: path/to/file.ext
```language
<content>
```
'''

prompt_coding_agent_paper2code = '''You are a coding agent implementing a research paper. You are provided with the paper text.

{BENCHMARK_RULES}

{TOY_EXAMPLE}

---
PAPER TO IMPLEMENT:
{paper_text}

---
Generate a complete, runnable repository. Include reproduce.sh, README.md, and all source code. Output format:
FILE: path/to/file.ext
```language
<content>
```
'''


# =============================================================================
# REACT REFINEMENT PROMPT (iterative improvement based on paper + feedback)
# =============================================================================

prompt_react_refinement = '''You are a coding agent implementing a research paper. You previously generated code, but there are issues to fix.

{BENCHMARK_RULES}

{TOY_EXAMPLE}

---
PAPER TO IMPLEMENT (reference for correctness):
{paper_text}

---
CURRENT CODE (has issues):
{current_code}

---
OBSERVATION / FEEDBACK (what went wrong or what to improve):
{feedback}

---
Your task: Fix the code to address the feedback. Ensure it aligns with the paper's description.
Output the COMPLETE corrected repository. Use the same format:
FILE: path/to/file.ext
```language
<content>
```

Only output files that need changes; you may output all files for clarity. Preserve correct parts.'''


# =============================================================================
# LLM-BASED PAPER REPRODUCTION CHECK (verify code implements paper requirements)
# =============================================================================

prompt_paper_reproduction_check = '''You are an expert reviewer. Compare the research paper with the generated reproduction code and verify that all important aspects from the paper are correctly implemented.

PAPER (excerpt):
{paper_text}

GENERATED CODE:
{generated_code}

---
Your task: Identify what from the paper MUST be reproduced and check if the code does it correctly.

Consider:
1. **Core method/algorithm**: Is the main contribution (e.g. model architecture, training procedure) implemented?
2. **Hyperparameters**: Are key values (learning rate, batch size, etc.) present and correct?
3. **Data setup**: Dataset, preprocessing, splits as described?
4. **Evaluation**: Metrics, protocol, baselines?
5. **Experimental setup**: Hardware, training steps, checkpoints?
6. **reproduce.sh**: Does it run the full pipeline to produce the paper's results?

Output format (be concise, actionable):
- MISSING: <requirement from paper> — not found in code
- INCORRECT: <requirement> — present but wrong (e.g. wrong value, wrong logic)
- PARTIAL: <requirement> — partially implemented, needs completion
- OK: <requirement> — correctly implemented

If everything important is reproduced, respond with: "ALL_OK: Reproduction appears complete."

Otherwise list all MISSING, INCORRECT, and PARTIAL items. Focus on what a developer must fix.'''


# =============================================================================
# COMPATIBILITY CHECK PROMPT (LLM-based entity resolution)
# =============================================================================

prompt_pattern_compatibility_check = '''You are comparing two implementations of the same pattern in a cookbook graph. The cookbook stores REUSABLE patterns from multiple papers. When a new paper adds a pattern that already exists, we must decide: are they COMPATIBLE (same approach, keep existing) or INCOMPATIBLE (different approach, create variant).

EXISTING pattern "{pattern_name}" in cookbook (triplets):
{existing_triplets}

NEW pattern "{pattern_name}" from current paper (triplets):
{new_triplets}

COMPATIBLE means: same libraries (e.g. both use HuggingFace), same flow, same config approach. A developer could use either interchangeably.
INCOMPATIBLE means: different libraries (e.g. HuggingFace vs custom loader), different pipeline, different dependencies. These are distinct implementations that should both exist as variants.

Respond with exactly one line:
COMPATIBLE
or
INCOMPATIBLE

If INCOMPATIBLE, add a second line with a brief reason (e.g. "different data source: HuggingFace vs custom CSV loader").

Response:
'''


# =============================================================================
# PROMPT FOR INCREMENTAL GRAPH UPDATE (Adding new paper to existing graph)
# =============================================================================

prompt_cookbook_incremental = '''Objective: The main goal is to extract implementation knowledge from a NEW paper and identify how it relates to EXISTING knowledge in the cookbook graph, enabling incremental graph construction across multiple papers.

Guidelines for Incremental Cookbook Knowledge Graph Building:

Creating Nodes and Triplets: Use the same structured triplet format: "subject, relation, object". When extracting from a new paper, you must consider the existing graph concepts and decide whether to link to existing entities or create new ones.
Remember that you should break complex triplets into simple atomic triplets.
Length of your triplet should not be more than 7 words.
Remember that object and subject must be atomic units while relation can be more complex and long.

EXISTING CONCEPTS IN THE GRAPH (from previous papers):
{existing_concepts}

ENTITY RESOLUTION RULES:

When you encounter a concept, determine its relationship to existing graph:

1. EXACT MATCH - Link to existing entity, do not duplicate:
   If concept has same name and meaning as existing entity, reference the existing one.
   Example: If "Self-Attention" exists and new paper uses "Self-Attention", extract triplets using existing entity name.

2. VARIANT - Create new entity with link to parent:
   If concept is similar but distinct from existing entity, create new and link.
   Examples: 'Relative Self-Attention, variant of, Self-Attention'; 'RoBERTa, extends, BERT'.

3. NOVEL - Create new entity:
   If concept does not exist in graph, create new entity.
   Examples: 'new method X, is novel, true'; 'new technique Y, introduced by, this paper'.

4. CONFIRMATION - Strengthen existing knowledge:
   If new paper confirms existing knowledge, note the confirmation.
   Examples: 'gradient clipping, confirmed by, paper Y'; 'best practice X, also used in, paper Y'.

5. CONTRADICTION - Flag conflicting information:
   If new paper contradicts existing knowledge, extract both with context.
   Examples: 'learning rate, value in paper X, 0.001'; 'learning rate, value in paper Y, 0.0001'; 'learning rate values, conflict between, paper X and Y'.

6. REFINEMENT - Add more specific information:
   If new paper provides more detail about existing concept, add as refinement.
   Examples: 'warmup steps, value for base model, 4000'; 'warmup steps, value for large model, 8000'.

CONTEXT-AWARE EXTRACTION:

For hyperparameters with different values across papers, always include context:
Examples: 'batch size, value on ImageNet, 256'; 'batch size, value on CIFAR, 128'; 'learning rate, recommended for Transformer, 0.0001'.

KNOWLEDGE CATEGORIES (same as base extraction):
- Implementation components, hyperparameters, experimental setup
- Training pipeline, dependencies, errors and solutions
- Best practices, gotchas, performance results

Do not miss connections to existing concepts. If observation mentions a technique that exists in the graph, link to it rather than creating a duplicate.
Several triplets can be extracted about the same node, including links to existing entities.

Example of triplets you have extracted before: {example}

Observation: {observation}

Remember that triplets must be extracted in format: "subject_1, relation_1, object_1; subject_2, relation_2, object_2; ..."

Extracted triplets: '''


# =============================================================================
# PROMPT FOR REFINING COOKBOOK TRIPLETS (Merging new with existing)
# =============================================================================

prompt_cookbook_refining = """You will be provided with list of existing triplets and list of new triplets from a cookbook knowledge graph. Triplets are in the following format: "subject, relation, object".
The triplets denote implementation knowledge about research papers: hyperparameters, configurations, errors, solutions, best practices, etc. When new papers are added to the cookbook, some triplets from the list of existing triplets may need to be updated or replaced with new triplets.

REPLACEMENT RULES:

Sometimes there are no triplets to replace:
Example of existing triplets: "learning rate, value, 0.001"; "model, trained on, ImageNet"; "batch size, value, 32".
Example of new triplets: "dropout, rate, 0.1"; "optimizer, type, Adam".
Example of replacing: []. Nothing to replace here - these are different parameters.

Sometimes triplets should be replaced when they contain conflicting information about the SAME aspect:
Example of existing triplets: "warmup steps, recommended value, 4000".
Example of new triplets: "warmup steps, recommended value, 8000 for large models".
Example of replacing: [["warmup steps, recommended value, 4000" -> "warmup steps, recommended value, 8000 for large models"]]. The new triplet provides more specific/updated information.

CRITICAL: Triplets should ONLY be replaced if they contain redundant or conflicting information about the SAME aspect of an entity.

DO NOT REPLACE triplets if they provide DIFFERENT or COMPLEMENTARY information:
Example of existing triplets: "gradient clipping, prevents, NaN loss"; "gradient clipping, max norm, 1.0".
Example of new triplets: "gradient clipping, improves, training stability".
Example of replacing: []. Nothing to replace - these triplets describe different aspects of gradient clipping.

DO NOT REPLACE triplets about errors/solutions/best practices that are ADDITIVE:
Example of existing triplets: "NaN loss, caused by, exploding gradients".
Example of new triplets: "NaN loss, caused by, division by zero".
Example of replacing: []. Nothing to replace - both causes are valid and should be kept.

SPECIAL RULES FOR COOKBOOK KNOWLEDGE:

1. HYPERPARAMETERS: Replace only if same parameter, same context, different value (newer is better).
2. ERRORS: Never replace - errors accumulate. Multiple causes/solutions are valid.
3. BEST PRACTICES: Replace only if directly contradictory. Confirmations should ADD, not replace.
4. PERFORMANCE RESULTS: Replace if same model, same dataset, same metric - keep latest result.
5. DEPENDENCIES: Replace if same library with updated version number.

I repeat, do not replace triplets if they carry different type of information about entities!!! It is better to leave a triplet than to replace one that has important information. Do not state that triplet needs to be replaced if you are not sure!!!
####

Generate only replacing, no descriptions are needed.
Existing triplets: {ex_triplets}.
New triplets: {new_triplets}.
####
Warning! Replacing must be generated strictly in following format: [[outdated_triplet_1 -> actual_triplet_1], [outdated_triplet_2 -> actual_triplet_2], ...], you MUST NOT include any descriptions in answer.
Replacing: """


# =============================================================================
# PROMPT FOR ERROR & SOLUTION EXTRACTION (Specialized)
# =============================================================================

prompt_cookbook_errors = '''Objective: The main goal is to extract ERRORS, BUGS, and their SOLUTIONS from implementation text, code comments, issues, or documentation to build error-handling knowledge in the cookbook graph.

Guidelines for Error Knowledge Extraction:

Creating Nodes and Triplets: Use structured triplet format: "subject, relation, object". For errors, capture the complete chain: error identification, symptoms, causes, solutions, and prevention.
Remember to break complex error descriptions into atomic triplets.
Length of your triplet should not be more than 7 words.
Remember that object and subject must be atomic units while relation can be more complex and long.

ERROR KNOWLEDGE TO EXTRACT:

1. ERROR IDENTIFICATION:
   Examples: 'NaN loss, type, numerical error'; 'OOM error, type, memory error'; 'shape mismatch, severity, critical'.

2. SYMPTOMS (Observable indicators):
   Examples: 'NaN loss, symptom, loss becomes inf'; 'OOM error, symptom, CUDA out of memory'; 'convergence issue, symptom, loss oscillates'.

3. CAUSES (Root causes):
   Examples: 'NaN loss, caused by, exploding gradients'; 'NaN loss, caused by, division by zero'; 'OOM error, caused by, large batch size'.

4. SOLUTIONS (Fixes):
   Examples: 'NaN loss, solved by, gradient clipping'; 'OOM error, solved by, reduce batch size'; 'shape mismatch, solved by, check dimensions'.

5. PREVENTION (Best practices):
   Examples: 'gradient clipping, prevents, NaN loss'; 'input validation, prevents, shape mismatch'.

6. DEBUGGING STRATEGIES:
   Examples: 'NaN loss, debug by, print gradient norms'; 'shape error, debug by, add assertions'.

COMMON ERROR CATEGORIES:
- Numerical: NaN, Inf, exploding/vanishing gradients, overflow/underflow
- Shape/Dimension: tensor mismatches, wrong broadcasting, incorrect axis
- Memory: OOM, memory leaks, inefficient allocation
- Convergence: loss not decreasing, instability, mode collapse
- Data: wrong format, missing preprocessing, data leakage
- Compatibility: version mismatches, API changes

Do not miss connections between errors and their solutions. If observation describes a problem and fix, extract both: 'error X, has symptom, Y', 'error X, caused by, Z', 'error X, solved by, W'.
Several triplets can be extracted about the same error covering different aspects.

Example of triplets you have extracted before: {example}

Observation: {observation}

Remember that triplets must be extracted in format: "subject_1, relation_1, object_1; subject_2, relation_2, object_2; ..."

Extracted triplets: '''


# =============================================================================
# PROMPT FOR BEST PRACTICES EXTRACTION (Specialized)
# =============================================================================

prompt_cookbook_best_practices = '''Objective: The main goal is to extract BEST PRACTICES, RECOMMENDATIONS, and TIPS from implementation text to build practical guidance in the cookbook graph.

Guidelines for Best Practice Knowledge Extraction:

Creating Nodes and Triplets: Use structured triplet format: "subject, relation, object". For best practices, capture the practice, what it improves, what it prevents, and when it applies.
Remember to break complex recommendations into atomic triplets.
Length of your triplet should not be more than 7 words.
Remember that object and subject must be atomic units while relation can be more complex and long.

BEST PRACTICE KNOWLEDGE TO EXTRACT:

1. PRACTICE IDENTIFICATION:
   Examples: 'gradient clipping, category, optimization'; 'warmup, importance, essential'; 'mixed precision, category, training'.

2. BENEFITS (What it improves):
   Examples: 'gradient clipping, prevents, NaN loss'; 'warmup, improves, training stability'; 'mixed precision, speeds up, training by 2x'.

3. IMPLEMENTATION:
   Examples: 'gradient clipping, max norm, 1.0'; 'warmup, steps, 4000'; 'layer norm, placed before, attention'.

4. APPLICABILITY:
   Examples: 'gradient clipping, applies to, Transformer training'; 'warmup, required for, large learning rates'; 'mixed precision, requires, GPU with tensor cores'.

5. ANTI-PATTERNS (What NOT to do):
   Examples: 'default initialization, causes, slow convergence'; 'no warmup, causes, training instability'; 'large batch without scaling lr, causes, poor generalization'.

6. GOTCHAS & CAVEATS:
   Examples: 'attention mask, must be, additive not boolean'; 'positional encoding, must be, buffer not parameter'; 'batch norm, problematic with, small batches'.

CATEGORIES OF BEST PRACTICES:
- Initialization: weight init, bias init, layer-specific init
- Optimization: learning rate schedules, gradient clipping, warmup, optimizer choices
- Regularization: dropout, weight decay, data augmentation, early stopping
- Architecture: layer norm placement, residual connections, attention patterns
- Data handling: preprocessing, batching, data loading, caching
- Training: mixed precision, gradient accumulation, checkpointing, logging

Do not miss connections between practices and their effects. If observation describes a technique and its benefit, extract both: 'technique X, improves, Y', 'technique X, prevents, Z'.
Several triplets can be extracted about the same practice covering different aspects.

Example of triplets you have extracted before: {example}

Observation: {observation}

Remember that triplets must be extracted in format: "subject_1, relation_1, object_1; subject_2, relation_2, object_2; ..."

Extracted triplets: '''


# =============================================================================
# PROMPT FOR GRAPH MERGE CLASSIFICATION
# =============================================================================

prompt_cookbook_merge_classify = """You will be provided with a NEW triplet and a list of EXISTING triplets from a cookbook knowledge graph. Your task is to classify how the new triplet relates to the existing graph.

Triplets are in the following format: "subject, relation, object".

CLASSIFICATION RULES:

Classify the new triplet into ONE of these categories:

1. ADD - New triplet provides novel information not in existing graph.
   Example: Existing has "learning rate, value, 0.001", new is "dropout, rate, 0.1" → ADD (different parameter)

2. DUPLICATE - New triplet is semantically identical to existing triplet.
   Example: Existing has "Adam, type, optimizer", new is "Adam, is type, optimizer" → DUPLICATE

3. EXTEND - New triplet adds detail to existing concept.
   Example: Existing has "warmup, improves, stability", new is "warmup, steps, 4000" → EXTEND (adds specific value)

4. CONFIRM - New triplet confirms existing knowledge from different source.
   Example: Existing has "gradient clipping, prevents, NaN loss", new is "gradient clipping, prevents, NaN loss" from different paper → CONFIRM

5. CONFLICT - New triplet contradicts existing triplet about same aspect.
   Example: Existing has "learning rate, recommended, 0.001", new is "learning rate, recommended, 0.0001" → CONFLICT

6. REFINE - New triplet provides more specific version of existing general triplet.
   Example: Existing has "batch size, value, 32", new is "batch size, value for ImageNet, 256" → REFINE

####
New triplet: {new_triplet}
Existing triplets: {existing_triplets}
####
Warning! Classification must be generated strictly as: CATEGORY: reason
Classification: """


# =============================================================================
# PROMPT FOR COOKBOOK KNOWLEDGE RETRIEVAL
# =============================================================================

prompt_cookbook_retrieve = '''Objective: Given a question about implementing a research paper, identify which triplets from the cookbook knowledge graph are relevant to answer it.

Question: {question}

Available triplets in the cookbook graph:
{triplets}

Guidelines for Retrieval:

1. For "how to" questions: Look for triplets about implementation, configuration, hyperparameters.
2. For "what if X fails" questions: Look for triplets about errors, symptoms, causes, solutions.
3. For "best way to" questions: Look for triplets about best practices, recommendations, gotchas.
4. For "what value for X" questions: Look for triplets about hyperparameters, default values, ranges.

Select triplets that directly answer the question or provide necessary context.
Return triplet indices that are relevant.

Relevant triplet indices (comma-separated): '''


# =============================================================================
# =============================================================================
# ORIGINAL DETAILED VERSIONS (V1) - More verbose with explicit schemas
# =============================================================================
# =============================================================================

# =============================================================================
# V1: DETAILED COOKBOOK EXTRACTION PROMPT
# =============================================================================

prompt_cookbook_extraction_v1_detailed = '''Objective: Extract comprehensive implementation knowledge from academic paper text to create a structured "cookbook" knowledge graph that enables reproducing the research AND avoiding common pitfalls.

Repository URL: {repo_url}

Focus: Extract ACTIONABLE knowledge covering:
1. HOW to implement and reproduce the work
2. WHAT can go wrong and how to prevent/fix it
3. BEST PRACTICES learned from this implementation

=============================================================================
CATEGORY 1: IMPLEMENTATION COMPONENTS
=============================================================================
Extract models, architectures, modules, algorithms, and their relationships.

Entity types to identify:
- Model, Architecture, Module, Layer, Component
- Algorithm, Function, Procedure, Operation
- DataStructure, Tensor, Format

Relation patterns:
- "component X, implements, concept Y"
- "module X, extends, base module Y"
- "layer X, connected to, layer Y"
- "function X, computes, output Y"
- "architecture X, contains, N layers of type Y"
- "module X, variant of, standard module Y"

=============================================================================
CATEGORY 2: HYPERPARAMETERS & CONFIGURATIONS
=============================================================================
Extract ALL numerical settings, thresholds, parameters with their values.

Entity types to identify:
- Hyperparameter, ConfigSetting, Threshold, Schedule
- DefaultValue, RecommendedValue, Range

Relation patterns:
- "hyperparameter X, default value, Y"
- "parameter X, recommended range, Y to Z"
- "setting X, critical for, behavior Y"
- "parameter X, set to, Y for task Z"
- "learning rate, schedule, warmup then decay"
- "batch size, constrained by, GPU memory"

=============================================================================
CATEGORY 3: EXPERIMENTAL SETUP & DATA
=============================================================================
Extract datasets, hardware, preprocessing, and evaluation protocols.

Entity types to identify:
- Dataset, DataSplit, Preprocessing, Augmentation
- Hardware, GPU, Memory, ComputeRequirement
- Metric, EvaluationProtocol, Benchmark

Relation patterns:
- "model X, trained on, dataset Y version Z"
- "preprocessing, includes step, normalization"
- "training, requires, N GPUs with X GB memory"
- "evaluation, uses metric, Y"
- "dataset X, split ratio, train:val:test = A:B:C"

=============================================================================
CATEGORY 4: TRAINING PIPELINE & WORKFLOW
=============================================================================
Extract sequential procedures, checkpointing, and workflow dependencies.

Entity types to identify:
- TrainingStep, PipelineStage, Checkpoint, Workflow
- Initialization, Optimization, Inference

Relation patterns:
- "training pipeline, step N, operation X"
- "step X, must precede, step Y"
- "checkpoint, saved every, N iterations"
- "early stopping, triggered by, metric X not improving for Y epochs"
- "inference, requires, preprocessing step X"

=============================================================================
CATEGORY 5: DEPENDENCIES & REQUIREMENTS
=============================================================================
Extract libraries, frameworks, pretrained models, and prerequisites.

Entity types to identify:
- Library, Framework, Version, Dependency
- PretrainedModel, Checkpoint, Baseline

Relation patterns:
- "implementation, requires, library X version Y"
- "model X, initialized from, pretrained checkpoint Y"
- "method X, depends on, library Y"
- "baseline comparison, includes, method X"
- "compatibility, requires, CUDA version X"

=============================================================================
CATEGORY 6: COMMON ERRORS & DEBUGGING
=============================================================================
Extract problems that can occur and their solutions.

Entity types to identify:
- Error, Bug, Failure, Issue
- Symptom, Cause, RootCause
- Solution, Fix, Workaround
- DebugStrategy, DiagnosticTool

Relation patterns:
- "error X, has symptom, observable behavior Y"
- "error X, caused by, root cause Y"
- "error X, solved by, solution Y"
- "symptom X, indicates, possible error Y"
- "debugging X, check first, condition Y"
- "issue X, workaround, temporary fix Y"
- "error X, commonly occurs when, condition Y"

Common errors to look for:
- Numerical instability (NaN, Inf, exploding/vanishing gradients)
- Shape mismatches and dimension errors
- Memory issues (OOM, memory leaks)
- Convergence problems (loss not decreasing, oscillating)
- Data issues (wrong format, missing preprocessing)
- Version incompatibilities

=============================================================================
CATEGORY 7: BEST PRACTICES & RECOMMENDATIONS
=============================================================================
Extract proven techniques and recommendations from the paper.

Entity types to identify:
- BestPractice, Recommendation, Tip
- Technique, Strategy, Pattern

Relation patterns:
- "best practice X, improves, aspect Y"
- "recommendation X, prevents, problem Y"
- "technique X, speeds up, process Y by Z%"
- "practice X, essential for, achieving result Y"
- "tip X, applies when, condition Y"
- "strategy X, preferred over, alternative Y because Z"

=============================================================================
CATEGORY 8: GOTCHAS & NON-OBVIOUS PITFALLS
=============================================================================
Extract subtle issues that are easy to miss.

Entity types to identify:
- Gotcha, Pitfall, Caveat, Warning
- Assumption, Constraint, Limitation

Relation patterns:
- "gotcha X, affects, component Y"
- "pitfall X, occurs when, assumption Y violated"
- "warning X, important for, correct implementation of Y"
- "caveat X, applies to, use case Y"
- "assumption X, must hold for, method Y to work"
- "limitation X, restricts, applicability to Y"

Examples of gotchas:
- API differences between framework versions
- Implicit assumptions in the paper not stated explicitly
- Order-of-operations that matter but aren't obvious
- Default values that differ from paper's recommendations
- Preprocessing steps that are mentioned only in appendix

=============================================================================
CATEGORY 9: PERFORMANCE & RESULTS
=============================================================================
Extract metrics, comparisons, and efficiency measurements.

Entity types to identify:
- Result, Score, Metric, Measurement
- Comparison, Baseline, Improvement
- Efficiency, Speed, Memory, FLOPs

Relation patterns:
- "method X, achieves, score Y on benchmark Z"
- "method X, outperforms, baseline Y by Z points"
- "optimization X, reduces, training time by Y%"
- "technique X, reduces memory, from Y to Z GB"
- "result X, statistically significant, p < Y"

=============================================================================
TRIPLET FORMAT RULES
=============================================================================
- Format: "subject, relation, object"
- Keep triplets atomic and specific
- ALWAYS include units and scales (%, seconds, tokens, dimensions)
- Include version numbers when mentioned
- Preserve exact values from the paper
- Use consistent naming for entities across triplets

=============================================================================
CRITICAL EXTRACTION PRIORITIES
=============================================================================
1. Extract ALL numerical values with their context
2. Capture step-by-step procedures with explicit ordering
3. Note every "gotcha" or non-obvious detail
4. Link errors to their symptoms, causes, AND solutions
5. Connect best practices to the problems they prevent
6. Preserve exact names (datasets, models, methods) as written
7. Include both default values AND experimental variations

Example triplets from previous extractions: {example}

Text to process: {observation}

Remember: Someone should be able to use these triplets to:
- Configure and run the implementation correctly
- Avoid common mistakes and debug issues
- Understand why certain choices were made
- Reproduce the paper's results

Format: "subject_1, relation_1, object_1; subject_2, relation_2, object_2; ..."

Extracted triplets: '''


# =============================================================================
# V1: DETAILED INCREMENTAL UPDATE PROMPT
# =============================================================================

prompt_cookbook_incremental_v1_detailed = '''Objective: Extract implementation knowledge from a NEW paper and identify how it relates to EXISTING knowledge in the cookbook graph.


NEW Repository: {repo_url}

EXISTING GRAPH CONCEPTS (from previous papers):
{existing_concepts}

=============================================================================
EXTRACTION TASK
=============================================================================
Extract triplets from the new paper text, with special attention to:

1. NOVEL CONCEPTS: New entities not in the existing graph
   - Mark clearly as new additions
   - Relation: "concept X, is novel, true"

2. SHARED CONCEPTS: Concepts that match existing graph entities
   - Link to existing entities rather than creating duplicates
   - Relation: "new implementation X, implements existing, concept Y"
   - Relation: "paper X, also uses, existing concept Y"

3. EXTENSIONS & VARIANTS: Modifications of existing concepts
   - Relation: "method X, extends, existing method Y"
   - Relation: "method X, variant of, existing method Y"
   - Relation: "method X, differs from Y, in aspect Z"

4. CONFIRMATIONS: Evidence that strengthens existing knowledge
   - Relation: "paper X, confirms, best practice Y"
   - Relation: "paper X, also reports, error Y"
   - Relation: "evidence for X, increased by, paper Y"

5. CONTRADICTIONS: Information that conflicts with existing graph
   - Relation: "paper X, contradicts, claim Y"
   - Relation: "paper X, recommends, value A (vs existing B)"
   - Relation: "conflict between, paper X and Y, regarding Z"

6. REFINEMENTS: More specific information about existing concepts
   - Relation: "paper X, specifies, parameter Y = Z (for context W)"
   - Relation: "paper X, adds detail, to existing concept Y"

=============================================================================
ENTITY RESOLUTION GUIDELINES
=============================================================================
When you encounter a concept, determine if it matches an existing one:

EXACT MATCH (link, don't duplicate):
- Same name and same meaning → use existing entity
- Example: "Self-Attention" in new paper = existing "Self-Attention"

VARIANT (create new, link to parent):
- Similar but distinct → create new entity with "variant_of" relation
- Example: "Relative Self-Attention" → new entity, "variant_of, Self-Attention"

NOVEL (create new):
- Not in existing graph → create new entity
- Mark with "is novel, true" for tracking

=============================================================================
CONTEXT-AWARE EXTRACTION
=============================================================================
For hyperparameters and settings, always include context:
- "parameter X, value Y, for architecture Z"
- "parameter X, value Y, on dataset Z"
- "parameter X, value Y, recommended by paper Z"

This allows the graph to store multiple valid values for different contexts.

=============================================================================
PROVENANCE TRACKING
=============================================================================
For every triplet, the paper source is implicit. When extracting:
- Include paper-specific details that might differ across papers
- Note when something is "paper-specific" vs "general"

Text to process: {observation}

Existing concepts for reference: {existing_concepts}

Example triplets: {example}

Format: "subject_1, relation_1, object_1; subject_2, relation_2, object_2; ..."

Extracted triplets with relationship to existing graph: '''


# =============================================================================
# V1: DETAILED ERROR EXTRACTION PROMPT
# =============================================================================

prompt_cookbook_errors_v1_detailed = '''Objective: Extract ERRORS, BUGS, and their SOLUTIONS from implementation text, code comments, issues, or documentation.

Source: {source_description}
Repository: {repo_url}

=============================================================================
ERROR EXTRACTION SCHEMA
=============================================================================

For each error/issue found, extract triplets covering:

1. ERROR IDENTIFICATION
   - "error X, type, category Y" (numerical/runtime/memory/convergence/data)
   - "error X, severity, level Y" (critical/high/medium/low)
   - "error X, frequency, Y" (very common/common/rare)

2. SYMPTOMS (Observable indicators)
   - "error X, symptom, observable Y"
   - "symptom Y, appears as, message or behavior Z"
   - "symptom Y, occurs during, phase Z"

3. CAUSES (Root causes)
   - "error X, caused by, root cause Y"
   - "cause Y, occurs when, condition Z"
   - "cause Y, related to, component Z"

4. SOLUTIONS (Fixes)
   - "error X, solved by, solution Y"
   - "solution Y, implementation, code or action Z"
   - "solution Y, applies to, context Z"

5. PREVENTION (Best practices that prevent the error)
   - "error X, prevented by, practice Y"
   - "practice Y, should be applied, at stage Z"

6. DEBUGGING (How to diagnose)
   - "error X, debug by, strategy Y"
   - "diagnostic Y, checks for, condition Z"

=============================================================================
COMMON ERROR CATEGORIES TO LOOK FOR
=============================================================================

NUMERICAL ERRORS:
- NaN/Inf values in loss or gradients
- Exploding or vanishing gradients
- Numerical overflow/underflow
- Division by zero

SHAPE/DIMENSION ERRORS:
- Tensor shape mismatches
- Incorrect broadcasting
- Wrong axis for operations

MEMORY ERRORS:
- Out of memory (OOM)
- Memory leaks
- Inefficient memory usage

CONVERGENCE ERRORS:
- Loss not decreasing
- Training instability
- Mode collapse (GANs)
- Overfitting/underfitting

DATA ERRORS:
- Wrong data format
- Missing preprocessing
- Data leakage
- Incorrect normalization

COMPATIBILITY ERRORS:
- Version mismatches
- API changes between versions
- Hardware incompatibilities

Text to process: {observation}

Example triplets: {example}

Format: "subject_1, relation_1, object_1; subject_2, relation_2, object_2; ..."

Extracted error-related triplets: '''


# =============================================================================
# V1: DETAILED BEST PRACTICES PROMPT
# =============================================================================

prompt_cookbook_best_practices_v1_detailed = '''Objective: Extract BEST PRACTICES, RECOMMENDATIONS, and TIPS from implementation text.

Source: {source_description}
Repository: {repo_url}

=============================================================================
BEST PRACTICE EXTRACTION SCHEMA
=============================================================================

For each best practice found, extract triplets covering:

1. PRACTICE IDENTIFICATION
   - "practice X, category, Y" (initialization/optimization/data/architecture/debugging)
   - "practice X, importance, level Y" (essential/recommended/optional)

2. DESCRIPTION & IMPLEMENTATION
   - "practice X, description, what to do Y"
   - "practice X, implementation, how to do Y"
   - "practice X, code pattern, snippet or approach Y"

3. BENEFITS
   - "practice X, improves, aspect Y"
   - "practice X, prevents, problem Y"
   - "practice X, speeds up, process Y"
   - "practice X, reduces, resource Y"

4. APPLICABILITY
   - "practice X, applies to, context Y"
   - "practice X, required for, architecture Y"
   - "practice X, especially important when, condition Y"

5. ANTI-PATTERNS (What NOT to do)
   - "anti-pattern X, description, what to avoid Y"
   - "anti-pattern X, causes, problem Y"
   - "practice Y, avoids, anti-pattern X"

6. EVIDENCE
   - "practice X, recommended by, paper Y"
   - "practice X, validated in, N papers"
   - "practice X, improves metric Y, by Z%"

=============================================================================
CATEGORIES OF BEST PRACTICES TO LOOK FOR
=============================================================================

INITIALIZATION:
- Weight initialization strategies
- Bias initialization
- Layer-specific initialization

OPTIMIZATION:
- Learning rate schedules
- Gradient clipping
- Warmup strategies
- Optimizer choices

REGULARIZATION:
- Dropout patterns
- Weight decay
- Data augmentation
- Early stopping criteria

ARCHITECTURE:
- Layer normalization placement
- Residual connections
- Attention patterns
- Activation functions

DATA HANDLING:
- Preprocessing pipelines
- Batching strategies
- Data loading optimization
- Caching strategies

TRAINING:
- Mixed precision training
- Gradient accumulation
- Checkpointing frequency
- Logging and monitoring

Text to process: {observation}

Example triplets: {example}

Format: "subject_1, relation_1, object_1; subject_2, relation_2, object_2; ..."

Extracted best practice triplets: '''


# =============================================================================
# V1: DETAILED MERGE DECISION PROMPT
# =============================================================================

prompt_cookbook_merge_decision_v1_detailed = '''Objective: Decide how to merge NEW triplets into an EXISTING cookbook knowledge graph.

EXISTING TRIPLETS (from previous papers):
{existing_triplets}

NEW TRIPLETS (from current paper):
{new_triplets}

=============================================================================
MERGE DECISION RULES
=============================================================================

For each new triplet, decide ONE of the following actions:

1. ADD: New triplet provides novel information
   - No similar triplet exists in the graph
   - Format: ["ADD", "new triplet"]

2. LINK: New triplet refers to existing concept
   - Same concept, can be connected
   - Format: ["LINK", "new triplet", "existing entity to link to"]

3. STRENGTHEN: New triplet confirms existing knowledge
   - Same information from different source (increases confidence)
   - Format: ["STRENGTHEN", "existing triplet", "evidence from new paper"]

4. EXTEND: New triplet adds detail to existing knowledge
   - More specific information about existing concept
   - Format: ["EXTEND", "existing triplet", "new detail triplet"]

5. CONFLICT: New triplet contradicts existing knowledge
   - Different values or claims for same thing
   - Format: ["CONFLICT", "existing triplet", "conflicting new triplet", "context"]

6. SUPERSEDE: New triplet replaces outdated information
   - Newer/better information available
   - Format: ["SUPERSEDE", "outdated triplet", "replacement triplet"]

7. SKIP: New triplet is redundant
   - Exact duplicate or semantically identical
   - Format: ["SKIP", "new triplet", "reason"]

=============================================================================
DECISION GUIDELINES
=============================================================================

- NEVER delete existing knowledge without strong justification
- Prefer EXTEND over SUPERSEDE when both could apply
- Mark CONFLICT rather than silently choosing one value
- LINK concepts across papers to build a connected graph
- STRENGTHEN when multiple papers agree (track evidence count)

For hyperparameters with different values:
- If different contexts (dataset, architecture) → ADD both with context
- If same context but different papers → CONFLICT (let user decide)
- If one is clearly more authoritative → SUPERSEDE with justification

For errors and solutions:
- Always ADD new errors (they accumulate)
- STRENGTHEN solutions that work across multiple papers
- EXTEND with additional causes or solutions

For best practices:
- STRENGTHEN practices confirmed by multiple papers
- ADD context-specific practices
- Note CONFLICTS in recommendations

=============================================================================
OUTPUT FORMAT
=============================================================================

Provide decisions as a list:
[
  ["ACTION", "triplet or reference", "additional info if needed"],
  ...
]

Merge decisions: '''


# =============================================================================
# V1: DETAILED QUERY GENERATION PROMPT
# =============================================================================

prompt_cookbook_query_v1_detailed = '''Given a user question about implementing a research paper, generate SPARQL-like queries to retrieve relevant information from the cookbook knowledge graph.

User Question: {question}

Available entity types in the graph:
- Paper, Concept, Algorithm, Module, Function
- Hyperparameter, ConfigSetting, DefaultValue
- Dataset, Metric, Hardware, Dependency
- Error, Symptom, Cause, Solution
- BestPractice, AntiPattern, Gotcha, Tip
- TrainingStep, Pipeline, Workflow

Available relation types:
- implements, extends, uses, requires, depends_on
- has_parameter, default_value, recommended_range
- trained_on, evaluated_on, achieves
- has_symptom, caused_by, solved_by, prevented_by
- applies_to, improves, prevents
- step_N, precedes, follows
- variant_of, confirmed_by, conflicts_with

Generate queries to answer the question:

Query patterns: '''
