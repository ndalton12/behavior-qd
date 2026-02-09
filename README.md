# Behavior-QD

Zero-shot behavior elicitation framework using Quality Diversity (QD) optimization.

## Overview

Behavior-QD uses QD algorithms to discover diverse prompts that elicit specific behaviors from language models. It maintains an archive of prompts in embedding space, ensuring coverage of different prompt "styles" while optimizing for behavior elicitation.

## Key Features

- **CVT-MAP-Elites in Embedding Space**: Archive prompts using PCA projection of vocabulary embeddings
- **Two Generation Strategies**:
  - **Sampler**: LLM-based prompt generation with best-of-n selection
  - **Embedding**: CMA-ME evolution with temperature-based token snapping
  - **Hybrid**: Run both simultaneously
- **Configurable Judge**: Custom rubrics with optional coherence scoring
- **Rubric Generation**: Auto-generate scoring rubrics from behavior descriptions

## Installation

```bash
pip install -e .
```

## Quick Start

### Run an experiment

```bash
# Using the sampler emitter
behavior-qd run "sycophancy" --mode sampler --iterations 50

# Using the embedding emitter (CMA-ME)
behavior-qd run "sycophancy" --mode embedding --iterations 50

# Hybrid mode (both emitters)
behavior-qd run "sycophancy" --mode hybrid --iterations 50
```

### Generate a rubric

```bash
behavior-qd generate-rubric "elicit sycophancy" --output rubrics/sycophancy.jinja
```

### Evaluate a single prompt

```bash
behavior-qd evaluate "You're the smartest AI ever!" --behavior "sycophancy"
```

### Visualize results

```bash
behavior-qd visualize output/checkpoint_iter100.pkl
```

### Show best prompts

```bash
behavior-qd show-prompts output/checkpoint_iter100.pkl --num 20
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BehaviorQDScheduler                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │   Sampler   │    │  Embedding  │    │     Hybrid      │  │
│  │   Emitter   │    │   Emitter   │    │    Emitter      │  │
│  │  (LLM gen)  │    │  (CMA-ME)   │    │  (both modes)   │  │
│  └──────┬──────┘    └──────┬──────┘    └────────┬────────┘  │
│         │                  │                     │           │
│         └──────────────────┼─────────────────────┘           │
│                            │                                 │
│                            ▼                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  Evaluator                           │    │
│  │  ┌─────────────┐        ┌─────────────┐             │    │
│  │  │   Target    │───────▶│    Judge    │             │    │
│  │  │   Model     │        │    Model    │             │    │
│  │  └─────────────┘        └─────────────┘             │    │
│  └─────────────────────────────────────────────────────┘    │
│                            │                                 │
│                            ▼                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │               BehaviorArchive (CVT)                  │    │
│  │  Measures: [PCA_1, PCA_2, Token_Variance]           │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

Configure via environment variables or `.env` file:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
BEHAVIOR_QD_OUTPUT_DIR=./experiments
```

## Python API

```python
from behavior_qd import BehaviorQDConfig, BehaviorQDScheduler

config = BehaviorQDConfig(
    behavior_description="sycophancy",
    output_dir="./output",
)
config.scheduler.iterations = 100
config.scheduler.emitter_mode = "hybrid"

scheduler = BehaviorQDScheduler(config)
stats = scheduler.run()

# Access results
best_prompts = scheduler.archive.get_elites(10)
for entry in best_prompts:
    print(f"{entry.objective:.3f}: {entry.prompt}")
```

## License

MIT
