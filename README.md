# PromptBERT Reimplementation

Reimplementation of [PromptBERT: Improving BERT Sentence Embeddings with Prompts](https://aclanthology.org/2022.emnlp-main.603/) (Jiang et al., EMNLP 2022).

Original repository: https://github.com/kongds/Prompt-BERT

## Modifications Made

Two changes were made to the original codebase:
1. `SentEval/senteval/sts.py`: Added `dtype=object` to `np.array()` calls for compatibility with `numpy>=1.24`
2. `requirements.txt`: Updated package versions for Python 3.12 compatibility

## Running Locally

**Requirements:** Python 3.12, pip

**Step 1: Clone the repository**
```bash
git clone https://github.com/AbdullahShheik/promptbert-reimplementation.git
cd promptbert-reimplementation
```

**Step 2: Create and activate a virtual environment**
```bash
python3 -m venv env
source env/bin/activate
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Download SentEval evaluation data**
```bash
cd SentEval/data/downstream
bash download_dataset.sh
cd -
```

**Step 5: Run evaluations**
```bash
# Evaluate all four pretrained model variants
bash eval_only.sh unsup-bert
bash eval_only.sh unsup-roberta
bash eval_only.sh sup-bert
bash eval_only.sh sup-roberta

# Analysis experiments
bash run.sh bert-prompt

# Static embedding bias removal experiments
bash run.sh roberta-base-static-embedding-remove-baises
bash run.sh bert-base-cased-static-embedding-remove-baises
bash run.sh bert-base-uncased-static-embedding-remove-baises
```

No random seed is needed for evaluation of pretrained checkpoints. The pretrained checkpoints are downloaded automatically from HuggingFace.

## Running on Kaggle

A ready-to-run Kaggle notebook is provided in the `Notebooks/` folder.

**Step 1:** Upload `Notebooks/promptbert-reproduction.ipynb` to Kaggle or open it directly.

**Step 2:** Enable GPU — go to Notebook Settings → Accelerator → GPU T4.

**Step 3:** The notebook handles all setup automatically including:
- Cloning this repository
- Installing compatible package versions
- Uninstalling conflicting pre-installed packages (`jax`, `jaxlib`)
- Downloading SentEval data
- Running all evaluations

## Evidence

The `Notebooks/` folder contains the Kaggle notebook with full output logs of all evaluation runs. Results include Spearman correlation scores on all seven STS benchmarks for all four pretrained model variants, the unfine-tuned bert-prompt experiment, and the static embedding bias removal experiments.

## Citation

```bibtex
@inproceedings{jiang-etal-2022-promptbert,
    title = "{P}rompt{BERT}: Improving {BERT} Sentence Embeddings with Prompts",
    author = "Jiang, Ting and Jiao, Jian and Huang, Shaohan and Zhang, Zihan and Wang, Deqing and Zhuang, Fuzhen and Wei, Furu and Huang, Haizhen and Deng, Denvy and Zhang, Qi",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    year = "2022",
    pages = "8826--8837",
}
```