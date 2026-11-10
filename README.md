# Evaluating the Performance and Reliability of AI Models in CV Screening

This repository contains two parts:

1. **Recruiter app (`app.py`)**: a Streamlit page where recruiters upload CVs, write must-have and nice-to-have requirements, and get a ranked shortlist with evidence from each CV. It uses Qwen (`qwen/qwen3.8-27b` via Groq).
2. **Model evaluation (`eval_dataset.py`)**: tests free open-weight models on 10 labelled CV–JD pairs, each in English and Indonesian. It measures accuracy, consistency, grounding, fairness and robustness.

## Project structure

```
app.py                  recruiter app (Streamlit)
src/screening.py        app logic: read files, mask identity, call the model, verify evidence, assign category
src/cvscreen.py         shared helpers (JSON parsing, evidence check)
src/prompt_dataset.txt  prompt template used in the evaluation
eval_dataset.py         runs the evaluation and computes metrics
data/dataset/           10 test cases (cv_en, cv_id, jd) + labels.csv
data/pilot/             3 pilot cases (format check only)
data/variants/          fairness (male/female name) and robustness (keyword stuffing, prompt injection) variants
results_*/              raw model outputs, logs, and metric tables
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file (already in `.gitignore`):

```
GEMINI_API_KEY=...
GROQ_API_KEY=...
```

## Running the app

```bash
streamlit run app.py
```

The model only quotes evidence for each requirement, and the code decides the category. Good Fit means every must-have is verified, Potential Fit means at least half are, and No Fit means fewer than half. Quotes that are missing from the CV, or that come only from the summary or skills list, are not counted. Names and contact details are masked before the CV is sent to the model. The final decision stays with the recruiter.

## Running the evaluation

| `--tool` | Model | Provider |
|---|---|---|
| `gemma` | gemma-4-31b-it | Google AI Studio |
| `gptoss` | openai/gpt-oss-120b | Groq |
| `qwen` | qwen/qwen3.8-27b | Groq |

```bash
python eval_dataset.py run --tool qwen --pilot     # 1. pilot: format check on 3 cases
python eval_dataset.py run --tool qwen --runs 1    # 2. main test: 10 cases x EN/ID
python eval_dataset.py run --tool qwen --varian    # 3. fairness & robustness variants
python eval_dataset.py evaluate                    # compute metrics
```

`run_all.command` runs all three models in parallel. The results go to `results_dataset/`, and `hasil_perbandingan.xlsx` and `report.md` summarise them.

## Dataset

The test cases come from HuggingFace `cnamuangtoun/resume-job-description-fit` (test split, labels Good / Potential / No Fit). Personal data is anonymised, and Indonesian translations are added. See `data/dataset/SOURCE.md` for details.

## Limitations

- The sample is small (10 cases), so the results are indicative only.
- The dataset labels are noisy, the dataset has no stated license, and the cases reflect the US job market.
- Free-tier models and their limits may change over time, so the test dates are logged.

## AI usage note

The code and the Indonesian CV translations were produced with help from Claude (Anthropic). The evaluation design, label review and interpretation of results are the author's own.
