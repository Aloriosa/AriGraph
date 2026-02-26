"""
Specialized prompt for extracting implementation knowledge from research papers
to create a reproducible "cookbook" knowledge graph.
"""

prompt_extraction_reproduction = '''Objective: Extract actionable implementation knowledge from academic paper text to create a structured "cookbook" knowledge graph that enables reproducing the research implementation.

Focus: Extract practical details about HOW to implement and reproduce the work, not just WHAT the research is about.

Guidelines for Reproduction-Focused Knowledge Extraction:

1. IMPLEMENTATION COMPONENTS:
   - Extract models, architectures, modules, and their relationships
   - Identify specific algorithms, functions, and procedures
   - Capture code structure and organization
   - Note how components extend or modify existing methods
   Examples of relation patterns:
   - "method X, extends, method Y"
   - "method X, implements, algorithm Y"
   - "function X, computes, metric Y"

2. HYPERPARAMETERS & CONFIGURATIONS:
   - Extract ALL numerical settings, thresholds, and parameters with their values
   - Capture optimization settings (learning rates, batch sizes, epochs)
   - Note ranges, schedules, and default values
   - Include symbolic parameters and their typical ranges
   Examples of relation patterns:
   - "method X, uses parameter Y, value Z"
   - "training, uses learning rate, [specific value]"
   - "parameter X, default value, Y"
   - "parameter X, ranges from, Y to Z"

3. EXPERIMENTAL SETUP:
   - Datasets used for training/evaluation with versions
   - Hardware requirements and specifications
   - Batch sizes, sequence lengths, and other run-time settings
   - Train/validation/test splits
   Examples of relation patterns:
   - "method X, evaluated on, dataset Y"
   - "experiments, run on, hardware Y"
   - "evaluation, uses metric, Y"
   - "dataset X, split into, Y portions"

4. ALGORITHMS & PROCEDURES:
   - Step-by-step processes and workflows
   - Mathematical operations and computations
   - Training and inference procedures in sequence
   - Optimization strategies
   Examples of relation patterns:
   - "method X, step 1, action Y"
   - "procedure X, first does Y, then does Z"
   - "algorithm X, computed as, formula Y"
   - "process X, iterates for, N steps"

5. DEPENDENCIES & REQUIREMENTS:
   - Required libraries, frameworks, and their versions
   - Pretrained models and checkpoints
   - Baseline methods for comparison
   - Software and hardware prerequisites
   Examples of relation patterns:
   - "implementation, requires framework, X"
   - "method X, depends on, library Y"
   - "baseline includes, method X"
   - "pretrained model, sourced from, X"

6. PERFORMANCE METRICS & RESULTS:
   - Specific numerical results with units
   - Comparison numbers against baselines
   - Efficiency measurements (time, memory, FLOPs)
   - Statistical significance indicators
   Examples of relation patterns:
   - "method X, achieves score, Y on dataset Z"
   - "method X, reduces metric Y, by Z%"
   - "comparison shows, X outperforms Y, by Z points"
   - "speedup measured as, X times faster"

7. IMPLEMENTATION DETAILS:
   - Initialization strategies for weights and biases
   - Loss functions and their composition
   - Regularization techniques
   - Data preprocessing and augmentation steps
   Examples of relation patterns:
   - "weights X, initialized with, distribution Y"
   - "loss function, combines, term X and term Y"
   - "data preprocessing, includes, operation X"
   - "regularization, applies, technique X with weight Y"

Triplet Format Rules:
- Use format: "subject, relation, object"
- Keep triplets atomic and specific
- Prefer concrete values over vague descriptions
- ALWAYS include units and scales when mentioned (%, seconds, tokens, dimensions)
- Link components to their purposes and functions
- Preserve mathematical notation and formulas when they are implementation-relevant

Key Relation Types to Use:
- "X, implements, Y" (implementation choice)
- "X, extends, Y" (builds upon existing method)
- "X, uses parameter, Y" (hyperparameter)
- "X, set to, Y" (configuration value)
- "X, requires, Y" (dependency)
- "X, evaluated on, Y" (experimental setup)
- "X, achieves, Y" (performance result)
- "X, initialized with, Y" (initialization strategy)
- "X, compared to, Y" (baseline comparison)
- "X, step N, Y" (procedural step)
- "X, computed as, Y" (mathematical formula)
- "X, depends on, Y" (technical dependency)

CRITICAL EXTRACTION RULES:
- Extract ALL numbers with their context (what they measure)
- Include dataset names, model names, and method names exactly as written
- Capture step-by-step procedures with explicit ordering
- Link components to their functions and purposes
- Note when something is "similar to", "based on", or "extends" existing work
- Preserve version numbers, dimensions, and scales
- Extract both default values and experimental ranges

Example of triplets you have extracted before: {example}

Text to process: {observation}

Remember: Focus on HOW to implement and reproduce, not just understanding the research.
Someone should be able to use these triplets to configure, implement, and run the method.
Format: "subject_1, relation_1, object_1; subject_2, relation_2, object_2; ..."

Extracted triplets: '''


prompt_extraction_reproduction_with_repo = '''Objective: Extract actionable implementation knowledge from academic paper text to create a structured "cookbook" knowledge graph for reproducing the research implementation.

Research repository: {repo_url}

Focus: Extract practical details about HOW to implement and reproduce the work, with awareness that there is an associated codebase for reference.

Guidelines for Reproduction-Focused Knowledge Extraction:

1. CODE STRUCTURE & MODULES:
   - Identify components that would exist as files/modules in a codebase
   - Link algorithms to likely implementation patterns
   - Note data structures and their representations
   Examples of relation patterns:
   - "component X, likely implemented as, module type Y"
   - "algorithm X, requires, data structure Y"
   - "feature X, stored as, format Y"

2. HYPERPARAMETERS & CONFIGURATIONS:
   - Extract ALL settings that would appear in configuration files
   - Note default values, typical ranges, and recommended values
   - Include command-line arguments and config parameters
   Examples of relation patterns:
   - "hyperparameter X, default value, Y"
   - "parameter X, typically ranges from, Y to Z"
   - "config setting X, controls, behavior Y"
   - "parameter X, must be set to, Y for task Z"

3. TRAINING PIPELINE & WORKFLOW:
   - Extract the sequence of operations in order
   - Identify checkpointing, logging, and evaluation steps
   - Note dependencies between steps
   Examples of relation patterns:
   - "training step N, performs, operation X"
   - "before step X, must complete, step Y"
   - "evaluation, performed every, N iterations"
   - "checkpoint saved, at interval, X steps"

4. DATA PREPARATION & PROCESSING:
   - Datasets with versions and splits
   - Preprocessing steps and transformations
   - Input/output formats and dimensions
   Examples of relation patterns:
   - "dataset X, version, Y"
   - "data preprocessing, step N, operation X"
   - "input format, requires dimensions, X by Y"
   - "dataset X, evaluation metric, Y"

5. MODEL ARCHITECTURE & OPERATIONS:
   - Layer structures and their connections
   - Mathematical operations with implementation hints
   - Forward/backward pass details
   Examples of relation patterns:
   - "layer X, connected to, layer Y"
   - "operation X, computed as, formula Y"
   - "architecture contains, N layers of type X"
   - "computation requires, operation X followed by Y"

6. BASELINES & EVALUATION:
   - Methods used for comparison
   - Evaluation protocols and metrics
   - Comparison settings and fairness criteria
   Examples of relation patterns:
   - "baseline method X, configuration, Y"
   - "comparison uses metric, X"
   - "evaluation protocol, follows, standard Y"
   - "fair comparison requires, setting X to Y"

7. IMPLEMENTATION-SPECIFIC DETAILS:
   - Framework-specific requirements and patterns
   - Optimization tricks and best practices
   - Common pitfalls and their solutions
   Examples of relation patterns:
   - "implementation framework, requires, library X"
   - "efficient implementation, uses technique, X"
   - "operation X, implemented via, framework feature Y"
   - "performance optimization, applies, trick X"

Triplet Format Rules:
- Format: "subject, relation, object"
- Prioritize implementation-actionable relations
- Include specific values, names, and versions
- Link to concrete, reproducible actions

Key Relations for Reproduction with Codebase Context:
- "X, implemented as, Y"
- "X, configured via, Y"
- "X, requires setting, Y"
- "X, trained on, Y"
- "X, evaluated using, Y"
- "X, compared against, Y"
- "pipeline step N, performs, X"
- "hyperparameter X, config key, Y"
- "module X, depends on, Y"
- "script X, runs, operation Y"

Repository Context: {repo_url}

Example of triplets you have extracted before: {example}

Text to process: {observation}

Focus on extracting knowledge that helps someone using the associated codebase:
1. Configure the environment and dependencies correctly
2. Set hyperparameters via config files or arguments
3. Understand the training pipeline execution order
4. Prepare data in the expected format
5. Reproduce experiments with correct settings
6. Navigate and understand the codebase organization

Format: "subject_1, relation_1, object_1; subject_2, relation_2, object_2; ..."

Extracted triplets: '''
