# Supported Evaluation Workflow Strategies

This document outlines which strategies from the **Unified Evaluation Workflow** are natively supported by the HuggingFace `evaluate` library in its full installation. A strategy is considered "supported" only if it is provided out-of-the-box after installing the library—no custom modules or external integrations required.

**Note:** This harness is mentioned in the unified workflow as: *Evaluate*

---

## Phase 0: Provisioning (The Runtime)

### Step A: Harness Installation

#### ✅ **Strategy 1: PyPI Packages** - **SUPPORTED**
The `evaluate` library can be installed via PyPI:
```bash
pip install evaluate
```
Additional features can be installed with extras:
```bash
pip install evaluate[evaluator]  # For evaluator features with transformers
pip install evaluate[template]   # For creating new metrics
```

#### ✅ **Strategy 2: Git Clone** - **SUPPORTED**
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

#### ✅ **Strategy 2: API Provider Authentication** - **SUPPORTED**
When using the `Evaluator` class with transformers integration, users can configure API keys for commercial model providers through environment variables or credential files for remote inference (e.g., using HuggingFace Inference API endpoints via transformers pipelines).

#### ✅ **Strategy 3: Repository Authentication** - **SUPPORTED**
The library integrates with HuggingFace Hub for accessing models and datasets. Users can authenticate using:
```bash
huggingface-cli login
```
This enables access to gated models and private datasets through the `datasets` library integration.

---

## Phase I: Specification (The Contract)

### Step A: SUT Preparation

#### ✅ **Strategy 1: Model-as-a-Service (Remote Inference)** - **SUPPORTED**
The `Evaluator` class supports configuring HTTP endpoints and API wrappers through the transformers pipeline interface. Users can pass model identifiers that point to remotely hosted models accessible via the HuggingFace Inference API.

#### ✅ **Strategy 2: Model-in-Process (Local Inference)** - **SUPPORTED**
The library fully supports loading model weights and checkpoints for local inference through:
- The `Evaluator` class with transformers integration
- Support for PyTorch, TensorFlow models via transformers pipelines
- Direct model loading for various tasks (text classification, question answering, ASR, image classification, etc.)

Example supported tasks:
- Text classification
- Token classification  
- Question answering
- Text generation
- Text2text generation (translation, summarization)
- Image classification
- Audio classification
- Automatic speech recognition

#### ❌ **Strategy 3: Algorithm Implementation (In-Memory Structures)** - **NOT SUPPORTED**
While the library provides metrics that can evaluate outputs from ANN algorithms or ranking systems, it does not provide native instantiation of specialized data structures like FAISS indexes or BM25 implementations.

#### ❌ **Strategy 4: Policy/Agent Instantiation (Stateful Controllers)** - **NOT SUPPORTED**
The library does not provide native support for instantiating RL policies or autonomous agents.

### Step B: Benchmark Preparation (Inputs)

#### ✅ **Strategy 1: Benchmark Dataset Preparation (Offline)** - **SUPPORTED**
The library provides comprehensive dataset loading through the `datasets` library integration:
- Loading from HuggingFace Hub via dataset names
- Data splitting (train/validation/test)
- Automatic format handling for different evaluation tasks
- Support for custom dataset objects

The `Evaluator` class includes `load_data()` method that handles:
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

#### ❌ **Strategy 1: Judge Preparation** - **NOT SUPPORTED**
While the library can load and use pre-trained models for evaluation, it does not provide native fine-tuning capabilities for judge models or reward models. Users must fine-tune judges externally.

#### ✅ **Strategy 2: Ground Truth Preparation** - **SUPPORTED**
The library's dataset integration inherently loads ground truth labels and references as part of benchmark datasets. The `prepare_data()` method in evaluators extracts references from datasets:
```python
{"references": data[label_column]}
```

---

## Phase II: Execution (The Run)

### Step A: SUT Invocation

#### ✅ **Strategy 1: Batch Inference** - **SUPPORTED**
The library's primary execution mode is batch inference through the `Evaluator` class:
- Processes multiple input samples through a fixed SUT instance
- Supports various batch processing scenarios
- The `compute()` method runs inference over entire datasets

Example:
```python
evaluator.compute(
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

#### ✅ **Strategy 1: Deterministic Measurement** - **SUPPORTED**
The library provides extensive deterministic metrics out-of-the-box, including:

**Exact matching & equality checks:**
- `accuracy`
- `exact_match`
- `f1`, `precision`, `recall`

**Distance metrics:**
- `cer` (Character Error Rate)
- `wer` (Word Error Rate)
- `mae` (Mean Absolute Error)
- `mse` (Mean Squared Error)

**Token-based text metrics:**
- `bleu`, `sacrebleu`, `google_bleu`
- `rouge`
- `meteor`
- `chrf`
- `ter`
- `sari`

**Other deterministic metrics:**
- `confusion_matrix`
- `matthews_correlation`
- `pearsonr`, `spearmanr`
- `roc_auc`
- `brier_score`
- And 50+ more metrics

#### ✅ **Strategy 2: Embedding Measurement** - **SUPPORTED**
The library includes embedding-based metrics that require transformation into learned embedding space:

- **`bertscore`**: Uses contextualized BERT embeddings for semantic similarity
- **`comet`**: Neural similarity model for machine translation evaluation (requires `unbabel-comet`)
- **`mauve`**: Uses embeddings for text generation evaluation
- **`bleurt`**: Learned metric using BERT-based embeddings

#### ❌ **Strategy 3: Subjective Measurement** - **NOT SUPPORTED**
While the library can load models that could be used as judges, it does not provide native LLM-as-a-judge functionality or frameworks for subjective evaluation. Users must implement subjective evaluation logic externally.

#### ✅ **Strategy 4: Performance Measurement** - **SUPPORTED**
The `Evaluator` class automatically computes performance metrics during evaluation:

**Time costs:**
- `total_time_in_seconds`: Total inference runtime
- `samples_per_second`: Throughput measurement
- `latency_in_seconds`: Per-sample latency

Implemented in the `_compute_time_perf()` method and included in all `compute()` results.

**Note:** Memory, FLOPs, and energy consumption metrics are not natively supported.

### Step B: Collective Aggregation

#### ✅ **Strategy 1: Score Aggregation** - **SUPPORTED**
All metrics in the library inherently perform score aggregation across instances:
- Metrics compute aggregate statistics (mean, sum, etc.) over all predictions
- The `compute()` method returns aggregated metrics
- The `combine()` function allows combining multiple metrics
- Support for custom aggregation logic in metric implementations

Example:
```python
metric = evaluate.load("accuracy")
results = metric.compute(predictions=preds, references=refs)  # Returns aggregate
```

#### ✅ **Strategy 2: Uncertainty Quantification** - **SUPPORTED**
The `Evaluator` class provides bootstrap-based confidence interval estimation:

```python
evaluator.compute(
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

**Requirements:** `scipy>=1.7.1` (included with `evaluate[evaluator]`)

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
While the library provides Gradio integration for creating metric demos (`evaluate.utils.gradio`), it does not provide comprehensive dashboard creation for displaying evaluation results, metric comparisons, or ranked tables.

#### ❌ **Strategy 5: Leaderboard Publication** - **NOT SUPPORTED**
The library does not provide native leaderboard submission or publication capabilities.

#### ❌ **Strategy 6: Regression Alerting** - **NOT SUPPORTED**
The library does not provide automated regression detection, baseline comparison, or alerting functionality.

---

## Summary

### Supported Strategies by Phase

**Phase 0: Provisioning**
- ✅ 2 out of 8 strategies supported (25%)

**Phase I: Specification**
- ✅ 4 out of 9 strategies supported (44%)

**Phase II: Execution**
- ✅ 1 out of 4 strategies supported (25%)

**Phase III: Assessment**
- ✅ 5 out of 6 strategies supported (83%)

**Phase IV: Reporting**
- ❌ 0 out of 6 strategies supported (0%)

### Overall Support
**✅ 12 out of 33 total strategies supported (36%)**

### Library Strengths

The `evaluate` library excels at:

1. **Metric Computation**: Extensive collection of 50+ metrics covering deterministic, embedding-based, and performance measurements
2. **Dataset Integration**: Seamless loading and preparation of benchmark datasets via HuggingFace Hub
3. **Model Evaluation**: Native support for evaluating both local and remote models through transformers integration
4. **Uncertainty Quantification**: Bootstrap-based confidence interval estimation
5. **Easy Installation**: Simple PyPI-based installation with optional extras

### Library Limitations

The library does not natively support:

1. **Interactive Evaluation**: No support for RL environments, multi-agent scenarios, or stateful execution
2. **Synthetic Data Generation**: No built-in data augmentation or perturbation
3. **Subjective Evaluation**: No LLM-as-a-judge or model-based subjective scoring framework
4. **Visualization & Reporting**: No charts, dashboards, or leaderboard integration
5. **Production Monitoring**: No streaming, regression detection, or alerting

The `evaluate` library is primarily designed as a **metric computation and batch evaluation framework** for standard ML/NLP tasks, with strong integration into the HuggingFace ecosystem.
