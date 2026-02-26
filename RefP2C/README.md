# RefP2C: Reflective Paper-to-Code Development Enabled by Fine-Grained Verification

Welcome to the RefP2C project! This repository contains our reflective paper-to-code reproduction framework. The system is divided into three core, independent workflows: **Initial Code Generation**, **Supervisory Signal Design**, and iterative **Code Reflection**.

---

## 📂 Project Structure

Here is an overview of the main directories and the roles they play within the RefP2C framework:

```
RefP2C/
├── model/                # Stores local embedding models for offline retrieval.
├── paper/                # Contains the source documents (e.g., paper.md) for processing.
├── results/              # Master directory for all generated outputs from the pipelines.
├── scripts/              # High-level entry points for running the different pipelines.
│   ├── generate_initial_code.py
│   ├── design_signals.py
│   ├── reflect_code.py
│   └── run.sh
└── src/                  # Houses all core application logic, organized into packages.
│   ├── core/       # Specialists for the Initial Code Generation pipeline.
│   ├── signals/          # Specialists for the Supervisory Signal Design pipeline.
│   ├── reflection/       # Specialists for the Code Reflection pipeline.
│   ├── prompts/          # Externalized LLM prompt templates.
└── └── utils/            # Shared, low-level utility modules (e.g., parsers).
```

---
