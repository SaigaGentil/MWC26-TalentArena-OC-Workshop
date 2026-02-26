# Openchip MWC26 TalentArena Workshop: Build GPT From Scratch

This repository contains the hands-on material for Openchip's MWC26 TalentArena workshop.

The goal is to help beginners understand how a GPT-style model is built step by step:
- full GPT2 architecture
- loading pre-trained weights into the created custom model
- autoregressive text generation

## Who This Is For

This workshop is designed for learners in deep learning and LLM engineering.

You do not need to be an expert in Transformers before starting. The notebook includes conceptual explanations and equations for each block.

## Repository Contents

- `build_gpt.ipynb`: main workshop notebook (architecture + equations + generation walkthrough)
- `arch/`: modular implementation of model components
- `generation/`: GPT-2 weight loading + text generation utilities
- `main.py`: simple runnable demo using the modular code
- `init.sh`: project bootstrap script (installs `uv` if missing, then runs `uv sync`)

## Prerequisites

- Linux/macOS shell with `bash`
- Internet access (to install dependencies and download GPT-2 weights)


## Quick Start

1. From the project root, run:

```bash
./init.sh
```

2. Open `build_gpt.ipynb` in your preferred Jupyter environment (for example, VS Code Notebook or JupyterLab) and execute cells top-to-bottom.

Suggested flow for attendees:
1. Read each markdown explanation before running the corresponding code cell.
2. Focus on tensor shapes and comments/equations in the attention/block sections.
3. Run the generation cells at the end and experiment with `temperature` and `top_k`.

## Troubleshooting

- If `uv` is not found after installation, open a new terminal and run `./init.sh` again.
- First run can take longer because dependencies and model weights are downloaded.
