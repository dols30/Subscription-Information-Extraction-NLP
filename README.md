# Subscription information extraction with DistilBERT

A named-entity recognition project for extracting subscription details from natural-language descriptions. The notebook fine-tunes DistilBERT to identify service, amount, currency, billing cycle, billing date, trial end, reminder, and plan type.

The original trained model was integrated into the [Due iOS app](https://github.com/dols30/Due).

## Published on Kaggle

- [View the published notebook: NLP Subscription Information Extraction-DistilBERT](https://www.kaggle.com/code/bashcode223/nlp-subscription-information-extraction-distilbert?scriptVersionId=356506008)
- [Dataset: Due Subscription Information Extraction](https://www.kaggle.com/datasets/bashcode223/due-subscription-information-extraction)

## Files

- [Subscription_Information_Extraction_NLP.ipynb](Subscription_Information_Extraction_NLP.ipynb): preprocessing, fine-tuning, threshold selection, evaluation, and example inference.
- [Due-Subscription-Dataset.zip](Due-Subscription-Dataset.zip): raw annotated descriptions, sentence patterns, processed splits, tokenizer files, and recorded baseline predictions.

## Dataset

| Split | Descriptions | Source |
|---|---:|---|
| Training | 3,796 | Synthetic descriptions |
| Development | 62 | Separately authored descriptions |
| Final evaluation | 84 | Separately authored descriptions |

Training examples were generated from 58 subscription patterns and 20 unrelated-text patterns, deduplicated, and shuffled with seed 32602. The dataset contains 3,942 descriptions in total. Prices and subscriptions are hypothetical examples; no customer data was collected.

The notebook reconstructs character-span annotations from the raw CSV, checks dataset fingerprints and split overlap, and creates 17 BIO token labels.

## Method and results

DistilBERT base cased is fine-tuned for four epochs using seed 326, batch size 16, learning rate 3e-5, and weight decay 0.01. Development entity F1 selects the checkpoint; development scores also select a confidence threshold. Final evaluation uses exact entity labels and text boundaries.

The original Due v2 experiment achieved **97.34% entity F1** and **85.71% complete-entry accuracy (72/84 descriptions)**. Those are reference results on a small, already evaluated benchmark. The uploaded notebook calculates its own scores when run. Its DistilBERT v1 and GLiNER comparisons use recorded baseline predictions.

## Run locally

From the repository folder:

```bash
unzip Due-Subscription-Dataset.zip -d dataset
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab torch numpy pandas matplotlib packaging
python -m jupyter lab Subscription_Information_Extraction_NLP.ipynb
```

Choose this environment as the notebook kernel and run the cells in order. The notebook installs its compatible Transformers, Accelerate, and Hugging Face Hub dependencies when needed. It discovers the extracted `dataset/` folder automatically and writes results under `output/`.

## License

[Apache License 2.0](LICENSE).

Author: Dol Raj Bashyal
