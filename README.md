# LLM Evaluation Toolkit

A comprehensive, hands on toolkit for evaluating Large Language Models across 12 evaluation categories. Includes 12 Jupyter notebooks with runnable examples and an interactive Streamlit dashboard with metrics.

Everything runs on **free Groq API inference**. No GPU required.

## What is This?

This project provides practical, code first implementations of every major LLM evaluation technique. Each notebook is standalone and covers a specific evaluation domain with real metrics, real frameworks, and real outputs.

The Streamlit dashboard brings all 12 categories into a single interactive interface with live Groq powered demos, gauge visualizations, and side by side comparisons.

## Evaluation Categories

| # | Category | Notebook | Key Metrics |
|---|---|---|---|
| 1 | Text Generation | `text_generation.ipynb` | BLEU, ROUGE, METEOR, BERTScore, Distinct N, Repetition, Hallucination, Verbosity, LLM Judge |
| 2 | Summarization | `text_summarization.ipynb` | ROUGE, BERTScore, Compression, Faithfulness, Readability (textstat) |
| 3 | RAG Evaluation | `rag.ipynb` | RAGAS (Faithfulness, Context Precision/Recall, Answer Relevancy), Groq Judge fallback |
| 4 | Text to SQL | `text_to_sql.ipynb` | Execution Accuracy, Component Match, VES, Schema Linking, Error Categorization |
| 5 | Safety & Robustness | `safety_and_robustness_eval.ipynb` | Detoxify Toxicity, Bias (counterfactual fairness), Jailbreak Detection, TruthfulQA, Consistency |
| 6 | Emerging Paradigms | `emerging_eval_paradigms.ipynb` | LLM as Judge (rubric), Pairwise Ranking, Self Consistency, Synthetic Eval |
| 7 | Cost & Efficiency | `cost_and_efficiency_simple.ipynb` | Latency, Tokens/sec, Info Density, Quality per Dollar |
| 8 | Long Context | `long_context_eval.ipynb` | NIAH, KV Retrieval, Position Bias, Multi Hop Reasoning, Counting Accuracy |
| 9 | Information Extraction | `text_extraction.ipynb` | NER P/R/F1, Edit Distance, Relation Extraction, Schema Conformity, Table Extraction |
| 10 | Multi Modal | `multi_modal_eval.ipynb` | BLEU, ROUGE, METEOR, BERTScore, CLIP Score, VQA, OCR, SSIM |
| 11 | Text to Image | `text_to_image_simple.ipynb` | CLIP Score, FID, Inception Score, LLM Preference |
| 12 | General Frameworks | `general_frameworks.ipynb` | EleutherAI lm eval, HELM, DeepEval, HF Evaluate, PromptBench |

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/llm-evaluation.git
cd llm-evaluation
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Set Up API Keys

```bash
cp .env.example .env
```

Open `.env` and add your keys:

```
GROQ_API_KEY = gsk_your_key_here
HUGGINGFACE_TOKEN = hf_your_token_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com). The HuggingFace token is optional (only needed for gated models).

### 3. Run Notebooks

```bash
jupyter notebook notebooks/
```

Each notebook is standalone. Open any one and run all cells.

### 4. Run Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard will open at `http://localhost:8501`. Select an evaluation type from the sidebar, enter inputs, and click Run.

## Project Structure

```
llm-evaluation/
├── notebooks/
│   ├── text_generation.ipynb          # BLEU, ROUGE, METEOR, BERTScore, diversity, LLM judge
│   ├── text_summarization.ipynb       # Summarization, QA, translation, paraphrase, style transfer
│   ├── rag.ipynb                      # Retrieval + generation evaluation with RAGAS
│   ├── text_to_sql.ipynb              # Execution accuracy, VES, schema linking
│   ├── safety_and_robustness_eval.ipynb  # Detoxify, bias, jailbreak, TruthfulQA
│   ├── emerging_eval_paradigms.ipynb  # LLM judge, pairwise, self consistency, synthetic eval
│   ├── cost_and_efficiency_simple.ipynb  # Latency, cost, quality per dollar
│   ├── long_context_eval.ipynb        # NIAH, KV retrieval, multi hop, counting
│   ├── text_extraction.ipynb          # NER, relations, schema validation, table extraction
│   ├── multi_modal_eval.ipynb         # Captioning, VQA, OCR, CLIP, SSIM
│   ├── text_to_image_simple.ipynb     # CLIP score, FID, inception score
│   ├── general_frameworks.ipynb       # EleutherAI, HELM, DeepEval, HF Evaluate, PromptBench
│   ├── eleuther_results/              # Pre computed GPT2 benchmark results
│   ├── t2i_eval/                      # Sample images for text to image evaluation
│   └── ocr_test/                      # Sample images for OCR testing
├── streamlit_app.py                   # Interactive evaluation dashboard (2100+ lines)
├── requirements.txt                   # All Python dependencies
├── .env.example                       # Template for API keys
├── .gitignore
├── LICENSE
└── README.md
```

## Streamlit Dashboard Features

The dashboard provides a unified interface for all 12 evaluation categories:

**For each category you get:**
- Description of what it evaluates and when to use it
- Framework tags showing which tools are used
- Metrics guide explaining what each number means
- Tools & Frameworks detail (why and how each tool is used)
- Live evaluation with gauge dashboards and comparison bar charts
- Detailed breakdown table with scores and explanations

**Dashboard specific features:**
- Side by side model comparison (8B vs 70B) for cost efficiency
- Pairwise ranking and self consistency in LLM as Judge
- Needle in a Haystack visualization across positions
- Key value retrieval, multi hop reasoning, and counting tests
- Schema conformity and table extraction for info extraction
- Synthetic test case generation for automated eval pipelines
- RAGAS integration with Groq judge fallback

## Frameworks and Tools Used

| Framework | What it does |
|---|---|
| [Groq](https://groq.com) | Free, fast inference API for Llama models |
| [EleutherAI lm eval](https://github.com/EleutherAI/lm-evaluation-harness) | Standard academic benchmarks (MMLU, ARC, HellaSwag) |
| [Stanford HELM](https://crfm.stanford.edu/helm/) | Holistic evaluation across 42 scenarios |
| [DeepEval](https://github.com/confident-ai/deepeval) | Production eval with RAG, hallucination, bias metrics |
| [HuggingFace Evaluate](https://huggingface.co/docs/evaluate) | 50+ metrics: BLEU, ROUGE, BERTScore, METEOR, etc. |
| [RAGAS](https://github.com/explodinggradients/ragas) | RAG specific evaluation framework |
| [PromptBench](https://github.com/microsoft/promptbench) | Adversarial robustness testing |
| [Detoxify](https://github.com/unitaryai/detoxify) | Toxicity detection model |
| [Sentence Transformers](https://www.sbert.net/) | Semantic similarity via embeddings |
| [CLIP / open clip](https://github.com/mlfoundations/open_clip) | Image text alignment scoring |
| [spaCy](https://spacy.io/) | NER baseline |
| [textstat](https://github.com/textstat/textstat) | Readability metrics (Flesch, etc.) |
| [SacreBLEU](https://github.com/mjpost/sacrebleu) | Translation evaluation |

## Requirements

- Python 3.10+
- Free Groq API key (get one at [console.groq.com](https://console.groq.com))
- Optional: HuggingFace token for gated models
- No GPU required (all inference is via Groq API)
