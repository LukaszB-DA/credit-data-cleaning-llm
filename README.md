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
record linkage, a deterministic combinatorial solver) with a language model
(Gemini) exactly where rules genuinely fail.

> **Language note:** the prompts sent to the model and the `print()` messages
> are written in Polish — a deliberate choice made during development, not an
> oversight. Code comments are being filled in bilingually (PL/EN); the repo
> will ultimately have this bilingual README.

## Data

Since real bank data (e.g. the full Lending Club dataset) wasn't available in
a sufficiently "raw" form (available copies were already pre-encoded/cleaned,
with no free-text fields), I prepared:

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

**Data exploration:** before defining any cleaning rule, I explored the data
iteratively in Data Wrangler (VS Code/Jupyter) — filters, `groupby`,
ASC/DESC sorting — to quickly spot outliers, illogical values, and patterns
in the dirty columns, before that translated into concrete rules in code
(e.g. `pd.to_numeric(errors="coerce")` to detect unparseable values, regex
when extracting `grade`).

## Architecture — three layers

### Layer 1: deterministic code (zero LLM)

Handles everything solvable with rules/text matching:

| Problem | Method | Accuracy |
|---|---|---|
| Missing values (5 representations) | `.replace()` on markers + `pd.to_numeric(errors="coerce")` for detection | 100% |
| Decimal separator | `pd.to_numeric(errors="coerce")` | 100% |
| Physically impossible values | range rules | 100% |
| `home_ownership`, `purpose` (typos) | fuzzy matching (rapidfuzz) | 100% |
| Phone number | digit extraction + reformatting | 100% |
| Customer duplicates | blocking + comparison (recordlinkage) | 99.3% of pairs, 100% accurate |

### Layer 1.5: deterministic date solver (zero LLM) — discovered mid-project

This is the most important revision in this project, so I'll describe **how
I got there**, not just the end result — the process itself is more
valuable than the outcome.

Originally, dates (`application_date`, `issue_date`) were cleaned exclusively
by the LLM — measured accuracy 98.78%/99.52%. I applied a **self-validating
check** that requires no ground truth: it verifies whether `issue_date`
satisfies the business rule (1-30 days after `application_date`) that **the
model itself** was given in the prompt. Records violating that rule are
suspect by definition — no ground truth needed. This caught **1,493
records** out of 101,500.

Digging further: these "dirty" dates (US m/d/y, EU d/m/y, ISO, spelled-out
month) actually have a **very small, countable space of possible
interpretations** — at most 12 known formats × 12 formats for the two
fields. I applied a solver: it generates all candidates for both dates, filters
combinations by the rule "issue_date is 1-30 days after application_date",
and:

- **exactly 1 matching combination** → resolved deterministically, zero LLM
- **0 or 2+ combinations** → genuinely ambiguous, needs an additional signal

Result on the full dataset: **101,452 of 101,500 records (99.95%) resolved
fully deterministically**, verified against ground truth at **100.000%
accuracy** wherever the solver claimed confidence. Only **48 records**
(0.05%) remain genuinely ambiguous.

**The conclusion this forced:** the LLM wasn't needed at all to solve this
particular problem at this scale. The task had the structure of a closed
combinatorial problem with an explicit resolving rule — exactly where a
classical solver always beats a probabilistic model, because it applies the
rule with 100% consistency, while the LLM (even given the exact same rule in
the prompt) doesn't always.

### Layer 2: LLM (Gemini 3.5 Flash-Lite) — genuine escalation only

After introducing the solver, the LLM is only needed where the problem
**doesn't have** a closed, enumerable solution space:

- **`advisor_notes`** — free text, an unbounded number of typo variants,
  impossible to exhaust with rules.
- **~48 genuinely ambiguous dates** (0.05% of records) — the solver has no
  way to resolve these, because the information physically doesn't exist in
  the data.
- **Duplicate gray zone** — ~0.7% of pairs where the code isn't confident.

## Final results (101,500 records)

| Metric | Result |
|---|---|
| `application_date` | 99.999% |
| `issue_date` | 100.000% |
| Duplicates (code) | 1,466 pairs flagged as duplicate, 100% accurate |
| Duplicates (LLM, gray zone) | 17/17 correct verdicts (7 confirmed duplicates + 10 correctly ruled out) |
| Total duplicates flagged | 1,473 (1,466 + 7) |
| Model used | `gemini-3.5-flash-lite` |

**For context:** typical manual data-entry error rates reported in the
literature are 1-5% (Panko et al.), rising to 18-40% for more complex/
heterogeneous documents. A 99.999%/100.000% result clearly exceeds a
realistic manual-cleaning scenario — and, more importantly, exceeds the LLM
running without solver support (98.78%/99.52%).

### Diagnosis — how the nature of the errors changed along the way

Before the solver existed, I analyzed the LLM's own errors:

- **91.2%** of wrong dates were an exact day↔month swap (both ≤12) — even
  the contextual hint (issue_date) sometimes wasn't enough for the model to
  resolve it unambiguously, even though the rule **mathematically** pointed
  to a single correct interpretation (verified: some "swap errors" violated
  the 1-30 day rule blatantly, e.g. a 98-day gap instead of 9 — the model had
  everything it needed, and still got it wrong).
- I identified a systematic model weakness with the `MM-DD-YYYY` format
  (hyphen) — the month was recognized correctly, but the day was sometimes
  wrongly copied from the month.

It was precisely this observation — that the model *had* sufficient
information and still failed — that led me to build a solver instead of
further tweaking the prompt.

## Pipeline — how it works

The whole pipeline is **one file** (`clean_credit_data_llm.py`), running 5
steps sequentially in a single execution:

```
dirty_data.csv
      │
      ▼
STEP 1/5: deterministic layer (code)  ──────► layer1_cleaned.csv
      │
STEP 2/5: duplicate detection (recordlinkage)
      │            │
      │            └──► auto_duplicate.csv / auto_not_duplicate.csv
      │                          │
      └──► duplicate_candidates_gray.csv (gray zone, ~0.7% of pairs)
                  │
                  ▼
STEP 3/5: LLM layer (Gemini, async)  ──► layer2_dates_notes_full.csv
      │                                   + duplicate_verdicts_llm.csv
      │        (skippable via the SKIP_LLM_RERUN switch if results
      │         from a previous session already exist)
      ▼
STEP 4/5: date solver (deterministic)  ──► overwrites dates in layer2
      │        99.95% of records resolved with zero LLM
      ▼
STEP 5/5: final merge  ──────► final_cleaned_data.csv  (101,500 rows)
```

The LLM layer runs **asynchronously, in batches of 100 records**, with
bounded concurrency (semaphore), a rate limiter, automatic retries on errors
(including response-completeness validation — if the model returns a
different record count than it received, the batch is automatically
retried), and a resume mechanism that lets processing continue across
sessions without repeating already-completed work.

## Repository structure

```
├── clean_data.csv                    # ground truth (never seen by the LLM)
├── dirty_data.csv                    # input data with controlled dirt
├── final_cleaned_data.csv            # FINAL RESULT
├── dirt_log_ground_truth.csv         # log of every injected error (for audit)
├── duplicate_ground_truth.csv        # duplicate mapping (ground truth)
├── generate_clean_data.py            # ground-truth generator
├── dirty_data.py                     # controlled-dirt generator
├── clean_credit_data_llm.py          # MAIN PIPELINE (5 steps, 1 file)
└── data_cleaning_process.ipynb       # exploration and decision-making process
```

## What's next / possible extensions

- A separate check on whether the duplicate pairs found by code overlap in
  `id` (a small discrepancy of 1,473 vs. 1,483 was found during merging)
- Extend the dirtying process to the `email` field (noted as skipped)
- For the ~48 genuinely ambiguous dates: an additional resolving signal
  (e.g. majority-format statistics for the dominant format in a given source)

---

*Author: [LukaszB-DA](https://github.com/LukaszB-DA) · Portfolio project — ETL with LLM, large-scale bank data cleaning (Python + Gemini API).*
