# ETL with LLM — smart large-scale cleaning of bank data

🇬🇧 English | 🇵🇱 [Polski](README.pl.md)

## About the project

The second stage of my data-cleaning portfolio — after manual, classical
cleaning of credit data (Excel/Power Query + Python), this project shows
**deliberate** use of an LLM inside an ETL pipeline: not "throw everything at
the model because you can", but an architecture that uses cheap,
deterministic code wherever it's enough, and the LLM only where it actually
adds value.

**Goal:** clean 101,500 bank records (loans, Lending-Club-style) in one
automated run, combining classical techniques (pandas, regex, fuzzy matching,
record linkage) with a language model (Gemini) exactly where rules fail —
ambiguous date formats and free text.

> **Language note:** the prompts sent to the model and the `print()` messages
> are written in Polish — a deliberate choice made during development, not an
> oversight. Code comments are being filled in bilingually (PL/EN); the repo
> will ultimately have this bilingual README.

## Data

Since real bank data (e.g. the full Lending Club dataset) wasn't available in
a sufficiently "raw" form (available copies were already pre-encoded/cleaned,
with no free-text fields), I built:

- **`clean_data.csv`** — 100,000 synthetic records, statistically calibrated
  to real Lending Club distributions (loan amounts, interest rate correlated
  with risk grade, real installment formulas, etc.) — this is the ground
  truth, never seen by the model.
- **`dirty_data.csv`** — 101,500 rows (100k + 1,500 duplicate customers with
  variations), with controlled, fully logged "dirt": missing values in 5
  different representations, 5 date formats, comma/dot decimal separator,
  category variants, text typos, physically impossible values.

Because I generated the dirt myself, I have an exact ground truth — I can
precisely measure the effectiveness of every cleaning stage instead of just
assuming it "looks right".

**Data exploration:** before writing any cleaning rule, I explored the data
iteratively in Data Wrangler (VS Code/Jupyter) — filters, `groupby`,
ASC/DESC sorting — to quickly spot outliers, illogical values, and patterns
in the dirty columns, before that translated into concrete rules in code
(e.g. `pd.to_numeric(errors="coerce")` to detect unparseable values, regex
when extracting `grade`).

## Architecture — two layers

### Layer 1: code (deterministic, zero LLM)

Handles everything solvable with rules/text matching:

| Problem | Method | Accuracy |
|---|---|---|
| Missing values (5 representations) | `.replace()` on markers + `pd.to_numeric(errors="coerce")` for detection | 100% |
| Decimal separator | `pd.to_numeric(errors="coerce")` | 100% |
| Physically impossible values | range rules | 100% |
| `home_ownership`, `purpose` (typos) | fuzzy matching (rapidfuzz) | 100% |
| Phone number | digit extraction + reformatting | 100% |
| Customer duplicates | blocking + comparison (recordlinkage) | 99.3% of pairs, 100% accurate |

### Layer 2: LLM (Gemini 3.5 Flash-Lite) — escalation only

**Only** what layer 1 cannot resolve unambiguously is routed here:

- **Dates** (`application_date`, `issue_date`) — 5 input formats, day/month
  ambiguity. The model uses context (issue_date is always 1-30 days after
  application_date) to resolve ambiguous cases.
- **`advisor_notes`** — free text, typos, style normalization.
- **Duplicate gray zone** — ~0.7% of pairs where the code isn't confident.

**Why this split:** before reaching for an LLM, I measured a classical date
parser's accuracy (a list of known formats) — it came out to **~61% correct
dates**, of which 3,657 cases were *silent* day/month mix-ups (the parser
didn't error, it just returned the wrong date). That's what showed this
particular task genuinely needed an LLM.

## Final results (101,500 records)

| Metric | Result |
|---|---|
| `application_date` | 98.78% |
| `issue_date` | 99.52% |
| Duplicates (code) | 1,466 pairs flagged as duplicate, 100% accurate |
| Duplicates (LLM, gray zone) | 17/17 correct verdicts (7 confirmed duplicates + 10 correctly ruled out) |
| Total duplicates flagged | 1,473 (1,466 + 7) |
| Model used | `gemini-3.5-flash-lite`, one model throughout |

**For context:** typical manual data-entry error rates reported in the
literature are 1-5% (Panko et al.), rising to 18-40% for more complex/
heterogeneous documents. A 98.78%/99.52% result is comparable to or better
than a realistic manual-cleaning scenario, in a fraction of the time.

### Diagnosing the remaining date errors

I didn't stop at the headline metric — I checked the **nature** of the errors:

- **91.2%** of wrong dates are an exact day↔month swap (both ≤12) — even the
  contextual hint (issue_date) sometimes isn't enough to resolve it
  unambiguously.
- I also identified a systematic model weakness with the `MM-DD-YYYY` format
  (hyphen) — the month was recognized correctly, but the day was sometimes
  wrongly copied from the month.
- The remaining ~9% of errors (including a handful of `null`s despite
  unambiguous dates) are isolated cases with no shared pattern.

## Pipeline — how it works

The whole pipeline is **one file** (`clean_credit_data_llm.py`), running 4
steps sequentially in a single execution:

```
dirty_data.csv
      │
      ▼
STEP 1/4: deterministic layer (code)  ──────► layer1_cleaned.csv
      │
STEP 2/4: duplicate detection (recordlinkage)
      │            │
      │            └──► auto_duplicate.csv / auto_not_duplicate.csv
      │                          │
      └──► duplicate_candidates_gray.csv (gray zone, ~0.7% of pairs)
                  │
                  ▼
STEP 3/4: LLM layer (Gemini, async)  ──► layer2_dates_notes_full.csv
      │                                   + duplicate_verdicts_llm.csv
      ▼
STEP 4/4: final merge  ──────► final_cleaned_data.csv  (101,500 rows)
```

The LLM layer runs **asynchronously, in batches of 100 records**, with
bounded concurrency (semaphore), a rate limiter, automatic retries on errors
(including response-completeness validation — if the model returns fewer
records than it received, the batch is automatically retried), and a resume
mechanism that lets processing continue across sessions without repeating
already-completed work.

## Repository structure

```
├── clean_data.csv                    # ground truth (never seen by the LLM)
├── dirty_data.csv                    # input data with controlled dirt
├── final_cleaned_data.csv            # FINAL RESULT
├── dirt_log_ground_truth.csv         # log of every injected error (for audit)
├── duplicate_ground_truth.csv        # duplicate mapping (ground truth)
├── generate_clean_data.py            # ground-truth generator
├── dirty_data.py                     # controlled-dirt generator
├── clean_credit_data_llm.py          # MAIN PIPELINE (4 steps, 1 file)
└── data_cleaning_process.ipynb       # exploration and decision-making process
```

## What's next / possible extensions

- A separate check on whether the duplicate pairs found by code overlap in
  `id` (a small discrepancy of 1,473 vs. 1,483 was found during merging)
- Test the model's accuracy on the `MM-DD-YYYY` format with an added heuristic
- Extend the dirtying process to the `email` field (noted as skipped)

---

*Author: [LukaszB-DA](https://github.com/LukaszB-DA) · Portfolio project — ETL with LLM, large-scale bank data cleaning (Python + Gemini API).*
