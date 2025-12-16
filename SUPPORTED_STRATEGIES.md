# Supported Evaluation Workflow Strategies

This document outlines which strategies from the **Unified Evaluation Workflow** are supported by the HuggingFace `evaluate` library.

**Classification Framework:**

- ✅ **NATIVELY SUPPORTED**: Available immediately after `pip install evaluate` with minimal configuration (≤2 lines of code). No external dependencies beyond the base harness installation required.

- 🔌 **SUPPORTED VIA THIRD-PARTY INTEGRATION**: Requires installing ≥1 external package(s) (e.g., `pip install evaluate[evaluator]` for transformers integration) or external tools. Typically needs glue code (≤10 lines) and has documented integration patterns.

- ❌ **NOT SUPPORTED**: Strategy requires custom implementation or is not available through any documented integration.

**Note:** This harness is mentioned in the unified workflow as: *Evaluate*

---

## Phase 0: Provisioning (The Runtime)

### Step A: Harness Installation

#### ✅ **Strategy 1: PyPI Packages** - **NATIVELY SUPPORTED**
The `evaluate` library can be installed via PyPI:
```bash
pip install evaluate
```
Additional features can be installed with extras:
```bash
pip install evaluate[evaluator]  # For evaluator features with transformers
pip install evaluate[template]   # For creating new metrics
```

#### ✅ **Strategy 2: Git Clone** - **NATIVELY SUPPORTED**
The library can be installed from source:
```bash
git clone https://github.com/huggingface/evaluate
cd evaluate
pip install -e .
```

#### ❌ **Strategy 3: Container Images** - **NOT SUPPORTED**
The library does not provide prebuilt Docker or OCI container images.

#### ❌ **Strategy 4: Binary Packages** - **NOT SUPPORTED**
The library does not provide standalone executable binaries.

#### ❌ **Strategy 5: Node Package** - **NOT SUPPORTED**
The library is Python-based and does not provide a Node.js package.

### Step B: Service Authentication

#### ❌ **Strategy 1: Evaluation Platform Authentication** - **NOT SUPPORTED**
The library does not provide authentication flows for evaluation platform services or leaderboard submission APIs.

#### 🔌 **Strategy 2: API Provider Authentication** - **SUPPORTED VIA THIRD-PARTY INTEGRATION**
Requires `pip install evaluate[evaluator]` to enable the `Evaluator` class with transformers integration. Users can then configure API keys for commercial model providers through environment variables or credential files for remote inference (e.g., using HuggingFace Inference API endpoints via transformers pipelines).

**Integration requirements:**
- Install: `pip install evaluate[evaluator]`
- Configure API keys via environment variables
- Use transformers pipeline interface

#### ✅ **Strategy 3: Repository Authentication** - **NATIVELY SUPPORTED**
The library integrates with HuggingFace Hub for accessing models and datasets. Users can authenticate using:
```bash
huggingface-cli login
```
This enables access to gated models and private datasets through the `datasets` library integration (included in base installation).

---

## Phase I: Specification (The Contract)

### Step A: SUT Preparation

#### 🔌 **Strategy 1: Model-as-a-Service (Remote Inference)** - **SUPPORTED VIA THIRD-PARTY INTEGRATION**
Requires `pip install evaluate[evaluator]` for the `Evaluator` class with transformers pipeline interface. Users can pass model identifiers that point to remotely hosted models accessible via the HuggingFace Inference API.

**Integration requirements:**
- Install: `pip install evaluate[evaluator]`
- Configure model endpoints via transformers pipeline
- Minimal code example (4 lines):
```python
from evaluate import evaluator
eval = evaluator("text-classification")
results = eval.compute(model_or_pipeline="model-on-api", data="dataset")
```

#### 🔌 **Strategy 2: Model-in-Process (Local Inference)** - **SUPPORTED VIA THIRD-PARTY INTEGRATION**
Requires `pip install evaluate[evaluator]` for the `Evaluator` class with transformers integration. Supports loading model weights and checkpoints for local inference through:
- PyTorch, TensorFlow models via transformers pipelines
- Direct model loading for various tasks

**Integration requirements:**
- Install: `pip install evaluate[evaluator]`
- Install model backend (PyTorch/TensorFlow)

**Supported tasks:**
- Text classification, Token classification, Question answering
- Text generation, Text2text generation (translation, summarization)
- Image classification, Audio classification
- Automatic speech recognition

**Minimal code example (3 lines):**
```python
from evaluate import evaluator
results = evaluator("text-classification").compute(
    model_or_pipeline="distilbert-base-uncased", data="rotten_tomatoes")
```

#### ❌ **Strategy 3: Algorithm Implementation (In-Memory Structures)** - **NOT SUPPORTED**
While the library provides metrics that can evaluate outputs from ANN algorithms or ranking systems, it does not provide native instantiation of specialized data structures like FAISS indexes or BM25 implementations.

#### ❌ **Strategy 4: Policy/Agent Instantiation (Stateful Controllers)** - **NOT SUPPORTED**
The library does not provide native support for instantiating RL policies or autonomous agents.

### Step B: Benchmark Preparation (Inputs)

#### ✅ **Strategy 1: Benchmark Dataset Preparation (Offline)** - **NATIVELY SUPPORTED**
The library provides comprehensive dataset loading through the `datasets` library (included in base installation):
- Loading from HuggingFace Hub via dataset names
- Data splitting (train/validation/test)
- Automatic format handling for different evaluation tasks
- Support for custom dataset objects

**Minimal code example (2 lines):**
```python
from datasets import load_dataset
data = load_dataset("rotten_tomatoes", split="test")
```

With the Evaluator class (requires `evaluate[evaluator]`):
```python
evaluator.load_data(data="rotten_tomatoes", split="test")
```

#### ❌ **Strategy 2: Synthetic Data Generation (Generative)** - **NOT SUPPORTED**
The library does not provide native data generation, perturbation, or augmentation capabilities. Users must generate synthetic data externally.

#### ❌ **Strategy 3: Simulation Environment Setup (Simulated)** - **NOT SUPPORTED**
The library does not provide simulation environments for RL or interactive scenarios.

#### ❌ **Strategy 4: Production Traffic Sampling (Online)** - **NOT SUPPORTED**
The library does not provide production traffic sampling or real-time stream processing capabilities.

### Step C: Benchmark Preparation (References)

#### 🔌 **Strategy 1: Judge Preparation** - **SUPPORTED VIA THIRD-PARTY INTEGRATION**
The library supports **loading and configuring pre-trained judge models** through two mechanisms:

**1. Model-based metrics** (some require external packages):
- `bleurt`: Loads BLEURT checkpoints (requires `pip install evaluate[tests]` for bleurt dependencies)
- `comet`: Loads COMET models (requires `unbabel-comet` package)
- `bertscore`: Loads BERT-based models (requires `bert-score` package)
- `mauve`: Loads GPT-2 for text generation evaluation (requires `mauve-text` package)

**2. Custom judge models** (requires `evaluate[evaluator]`):
```python
# Requires: pip install evaluate[evaluator]
from evaluate import evaluator
results = evaluator("text-classification").compute(
    model_or_pipeline="your-judge-model",
    data=test_data,
    metric="accuracy"
)
```

**Not supported**: Fine-tuning discriminative models or reward models to create specialized judges. Users must fine-tune judges externally and then load them.

#### ✅ **Strategy 2: Ground Truth Preparation** - **NATIVELY SUPPORTED**
The library's dataset integration (included in base installation) inherently loads ground truth labels and references as part of benchmark datasets. 

**Minimal code example (2 lines):**
```python
from datasets import load_dataset
data = load_dataset("squad", split="validation")  # Includes ground truth answers
```

---

## Phase II: Execution (The Run)

### Step A: SUT Invocation

#### 🔌 **Strategy 1: Batch Inference** - **SUPPORTED VIA THIRD-PARTY INTEGRATION**
The library's primary execution mode is batch inference through the `Evaluator` class (requires `pip install evaluate[evaluator]`):
- Processes multiple input samples through a fixed SUT instance
- Supports various batch processing scenarios
- The `compute()` method runs inference over entire datasets

**Integration requirements:**
- Install: `pip install evaluate[evaluator]`

**Example (4 lines):**
```python
from evaluate import evaluator
results = evaluator("text-classification").compute(
    model_or_pipeline="distilbert-base-uncased",
    data="rotten_tomatoes",
    metric="accuracy"
)
```

#### ❌ **Strategy 2: Interactive Loop** - **NOT SUPPORTED**
The library does not provide native support for stateful step-by-step execution with environment interactions, tool-based reasoning, or multi-agent coordination.

#### ❌ **Strategy 3: Arena Battle** - **NOT SUPPORTED**
The library does not provide native support for pairwise model comparison where the same input is sent to multiple models for direct comparison.

#### ❌ **Strategy 4: Production Streaming** - **NOT SUPPORTED**
The library does not support continuous processing of live production traffic or real-time metric collection.

---

## Phase III: Assessment (The Score)

### Step A: Individual Scoring

#### ✅ **Strategy 1: Deterministic Measurement** - **NATIVELY SUPPORTED**
The library provides extensive deterministic metrics out-of-the-box (included in base installation):

**Exact matching & equality checks:**
- `accuracy`, `exact_match`, `f1`, `precision`, `recall`

**Distance metrics:**
- `cer` (Character Error Rate), `wer` (Word Error Rate)
- `mae` (Mean Absolute Error), `mse` (Mean Squared Error)

**Token-based text metrics:**
- `bleu`, `sacrebleu`, `google_bleu`, `rouge`, `meteor`, `chrf`, `ter`, `sari`

**Other deterministic metrics:**
- `confusion_matrix`, `matthews_correlation`, `pearsonr`, `spearmanr`, `roc_auc`, `brier_score`
- And 50+ more metrics

**Minimal code example (2 lines):**
```python
import evaluate
metric = evaluate.load("accuracy")
results = metric.compute(predictions=[0, 1], references=[0, 1])
```

#### 🔌 **Strategy 2: Embedding Measurement** - **SUPPORTED VIA THIRD-PARTY INTEGRATION**
The library includes embedding-based metrics that require external package installations:

- **`bertscore`**: Requires `pip install bert-score` - Uses contextualized BERT embeddings for semantic similarity
- **`comet`**: Requires `pip install unbabel-comet` - Neural similarity model for machine translation evaluation
- **`mauve`**: Requires `pip install mauve-text` - Uses embeddings for text generation evaluation
- **`bleurt`**: Requires bleurt dependencies - Learned metric using BERT-based embeddings

**Example (3 lines):**
```python
import evaluate
metric = evaluate.load("bertscore")
results = metric.compute(predictions=["hello"], references=["hi"], lang="en")
```

#### 🔌 **Strategy 3: Subjective Measurement** - **SUPPORTED VIA THIRD-PARTY INTEGRATION**
The library supports using pre-trained models as judges through model-based metrics (require external packages), but does not provide a comprehensive LLM-as-a-judge framework.

**Integration options:**
1. **Model-based metrics for subjective qualities** (each requires specific package):
   - `bleurt`: Learned metric for translation quality (requires bleurt dependencies)
   - `comet`: Neural model for MT evaluation with human-like judgments (requires `unbabel-comet`)
   - `bertscore`: Semantic similarity using contextualized embeddings (requires `bert-score`)
   - `mauve`: Text generation quality assessment (requires `mauve-text`)

2. **Custom judge models** (requires `evaluate[evaluator]`):
```python
# Requires: pip install evaluate[evaluator]
from evaluate import evaluator
results = evaluator("text-classification").compute(
    model_or_pipeline="reward-model-judge",
    data=test_data,
    metric="accuracy"
)
```

**Not supported**: 
- Native LLM-as-a-judge framework for prompting large language models to provide ratings
- Pairwise comparison frameworks (requires Phase II-A-3 Arena Battle, which is not supported)
- Built-in subjective evaluation workflows for attributes like helpfulness, harmlessness, or hallucination

Users must implement custom logic to use LLMs (via API or local inference) as judges for subjective evaluation.

#### 🔌 **Strategy 4: Performance Measurement** - **SUPPORTED VIA THIRD-PARTY INTEGRATION**
The `Evaluator` class (requires `pip install evaluate[evaluator]`) automatically computes performance metrics during evaluation:

**Integration requirements:**
- Install: `pip install evaluate[evaluator]`

**Time costs automatically measured:**
- `total_time_in_seconds`: Total inference runtime
- `samples_per_second`: Throughput measurement
- `latency_in_seconds`: Per-sample latency

Implemented in the `_compute_time_perf()` method and included in all `compute()` results.

**Note:** Memory, FLOPs, and energy consumption metrics are not supported.

### Step B: Collective Aggregation

#### ✅ **Strategy 1: Score Aggregation** - **NATIVELY SUPPORTED**
All metrics in the library inherently perform score aggregation across instances (included in base installation):
- Metrics compute aggregate statistics (mean, sum, etc.) over all predictions
- The `compute()` method returns aggregated metrics
- The `combine()` function allows combining multiple metrics
- Support for custom aggregation logic in metric implementations

**Minimal code example (2 lines):**
```python
import evaluate
metric = evaluate.load("accuracy")
results = metric.compute(predictions=preds, references=refs)  # Returns aggregate
```

#### 🔌 **Strategy 2: Uncertainty Quantification** - **SUPPORTED VIA THIRD-PARTY INTEGRATION**
The `Evaluator` class (requires `pip install evaluate[evaluator]`) provides bootstrap-based confidence interval estimation:

**Integration requirements:**
- Install: `pip install evaluate[evaluator]` (includes `scipy>=1.7.1`)

**Example:**
```python
from evaluate import evaluator
results = evaluator("text-classification").compute(
    model_or_pipeline=model,
    data=data,
    metric="accuracy",
    strategy="bootstrap",        # Enable uncertainty quantification
    confidence_level=0.95,       # Confidence level
    n_resamples=9999,           # Number of bootstrap samples
    random_state=42
)
```

This uses SciPy's `bootstrap` method to compute confidence intervals and standard errors for metric scores.

**Note:** Prediction-Powered Inference (PPI) is not supported.

---

## Phase IV: Reporting (The Output)

### Step A: Insight Presentation

#### ❌ **Strategy 1: Execution Tracing** - **NOT SUPPORTED**
The library does not provide detailed step-by-step execution logs showing tool calls, intermediate reasoning states, or agent decision paths.

#### ❌ **Strategy 2: Subgroup Analysis** - **NOT SUPPORTED**
The library does not provide native stratified analysis by demographic groups, domains, or task categories. Users must implement subgroup filtering and analysis externally.

#### ❌ **Strategy 3: Chart Generation** - **NOT SUPPORTED**
The library does not provide native visualization or chart generation capabilities. Users must use external libraries (matplotlib, plotly, etc.) for visualization.

#### ❌ **Strategy 4: Dashboard Creation** - **NOT SUPPORTED**
The library provides Gradio integration for creating metric demos (`evaluate.utils.gradio`). However, it does not provide comprehensive dashboard capabilities for displaying evaluation results, metric comparisons, or ranked tables.

#### ❌ **Strategy 5: Leaderboard Publication** - **NOT SUPPORTED**
The library does not provide native leaderboard submission or publication capabilities.

#### ❌ **Strategy 6: Regression Alerting** - **NOT SUPPORTED**
The library does not provide automated regression detection, baseline comparison, or alerting functionality.

---

## Summary

### Supported Strategies by Phase

**Phase 0: Provisioning**
- ✅ 2 natively supported, 🔌 1 via third-party integration, ❌ 5 not supported (2/8 native, 3/8 total = 37.5%)

**Phase I: Specification**
- ✅ 2 natively supported, 🔌 3 via third-party integration, ❌ 4 not supported (2/9 native, 5/9 total = 55.6%)

**Phase II: Execution**
- ✅ 0 natively supported, 🔌 1 via third-party integration, ❌ 3 not supported (0/4 native, 1/4 total = 25%)

**Phase III: Assessment**
- ✅ 2 natively supported, 🔌 4 via third-party integration, ❌ 0 not supported (2/6 native, 6/6 total = 100%)

**Phase IV: Reporting**
- ❌ 6 not supported (0/6 = 0%)

### Overall Support
**Total: 6 natively supported + 9 via third-party integration = 15/34 strategies supported (44% total support, 18% native)**

### Library Strengths

The `evaluate` library excels at:

1. **Metric Computation**: Extensive collection of 50+ metrics with native support for deterministic measurements and third-party integration for embedding-based metrics
2. **Dataset Integration**: Native seamless loading and preparation of benchmark datasets via HuggingFace Hub (included in base installation)
3. **Model Evaluation**: Third-party integration support for evaluating both local and remote models through transformers (requires `evaluate[evaluator]`)
4. **Judge Model Support**: Third-party integration for loading and using pre-trained judge models through model-based metrics (BLEURT, COMET, BERTScore) and custom evaluators
5. **Uncertainty Quantification**: Third-party integration for bootstrap-based confidence interval estimation (requires `evaluate[evaluator]`)
6. **Easy Installation**: Simple PyPI-based installation with optional extras for extended functionality

### Library Limitations

The library does not support:

1. **Interactive Evaluation**: No support for RL environments, multi-agent scenarios, or stateful execution
2. **Synthetic Data Generation**: No built-in data augmentation or perturbation
3. **Comprehensive Subjective Evaluation**: Third-party integration for model-based judges but lacks native LLM-as-a-judge framework and pairwise comparison workflows
4. **Judge Fine-tuning**: Cannot fine-tune judge models—users must train externally and load pre-trained judges
5. **Visualization & Reporting**: No charts, dashboards, or leaderboard integration
6. **Production Monitoring**: No streaming, regression detection, or alerting

### Architecture Notes

The `evaluate` library has a **two-tier architecture**:

1. **Base installation** (`pip install evaluate`): Provides core metric computation and dataset loading
   - Includes: `datasets`, `numpy`, `pandas`, `huggingface-hub`, and other utilities
   - Supports: Metric loading, dataset loading, ground truth preparation, score aggregation

2. **Evaluator extension** (`pip install evaluate[evaluator]`): Adds model evaluation capabilities
   - Requires: `transformers`, `scipy>=1.7.1`
   - Enables: Model loading (local/remote), batch inference, performance measurement, uncertainty quantification

The library is primarily designed as a **metric computation and batch evaluation framework** for standard ML/NLP tasks, with strong integration into the HuggingFace ecosystem.
