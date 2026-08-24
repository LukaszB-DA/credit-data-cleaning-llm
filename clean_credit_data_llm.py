"""
clean_credit_data_llm.py

PL: Kompletny pipeline czyszczenia danych bankowych (101 500 rekordów).
    Architektura po odkryciu solvera dat:
    (1) deterministyczny kod - pandas/regex/fuzzy matching/record linkage,
    (1.5) deterministyczny SOLVER DAT - generuje kandydatów z surowych dat
          (12 znanych formatów), filtruje regułą biznesową (issue_date jest
          1-30 dni po application_date), rozstrzyga 99,95% przypadków BEZ LLM,
    (2) LLM (Gemini) - tylko tam, gdzie warstwa 1/1.5 nie rozstrzyga
        jednoznacznie: advisor_notes (wolny tekst), garstka (~0,05%)
        naprawdę niejednoznacznych dat, strefa szara duplikatów.

EN: Full bank-data cleaning pipeline (101,500 records). Architecture after
    the date-solver discovery:
    (1) deterministic code - pandas/regex/fuzzy matching/record linkage,
    (1.5) deterministic DATE SOLVER - generates candidates from raw dates
          (12 known formats), filters by the business rule (issue_date is
          1-30 days after application_date), resolves 99.95% of cases with
          ZERO LLM calls,
    (2) LLM (Gemini) - only where layer 1/1.5 cannot resolve unambiguously:
        advisor_notes (free text), a handful (~0.05%) of genuinely ambiguous
        dates, gray-zone duplicate pairs.

Uruchomienie / Usage:
    python clean_credit_data_llm.py

Wymaga uzupełnienia klucza API poniżej / Requires the API key to be filled in below:
    GEMINI_API_KEY = "..." (patrz sekcja KONFIG / see CONFIG section)

Wynik / Output:
    final_cleaned_data.csv
"""

import asyncio
import json
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import recordlinkage
from google import genai
from pydantic import BaseModel
from rapidfuzz import fuzz, process

# ============================================================
# KONFIG / CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DIRTY_DATA_PATH = BASE_DIR / "dirty_data.csv"
LAYER1_OUTPUT = BASE_DIR / "layer1_cleaned.csv"
GRAY_ZONE_PATH = BASE_DIR / "duplicate_candidates_gray.csv"
AUTO_DUP_PATH = BASE_DIR / "auto_duplicate.csv"
AUTO_NOT_DUP_PATH = BASE_DIR / "auto_not_duplicate.csv"
LAYER2_OUTPUT = BASE_DIR / "layer2_dates_notes_full.csv"
DUPLICATE_VERDICTS_PATH = BASE_DIR / "duplicate_verdicts_llm.csv"
FINAL_OUTPUT = BASE_DIR / "final_cleaned_data.csv"

GEMINI_API_KEY = "API_KEY"
MODEL_NAME = "gemini-3.5-flash-lite"

CHUNK_SIZE = 100
MAX_CONCURRENT = 10
MAX_RETRIES = 3
RATE_LIMIT_CALLS_PER_MINUTE = 100

# PL: PRZELACZNIK - jesli masz juz layer2_dates_notes_full.csv i
#     duplicate_verdicts_llm.csv z wczesniejszego uruchomienia, ustaw True,
#     zeby NIE wysylac ponownie zadnych zapytan do Gemini. Solver dat i tak
#     poprawi 99,95% dat lokalnie, za darmo. Ustaw False tylko przy
#     zupelnie swiezym uruchomieniu od zera (bez istniejacych plikow LLM).
# EN: SWITCH - if you already have layer2_dates_notes_full.csv and
#     duplicate_verdicts_llm.csv from a previous run, set True to AVOID
#     sending any new requests to Gemini. The date solver will still fix
#     99.95% of dates locally, for free. Set False only for a completely
#     fresh run (no existing LLM output files).
SKIP_LLM_RERUN = False

MISSING_MARKERS = ["", "NULL", "N/A", "n/a", "unknown", "-1", "nan", "???", "no phone", "00000"]
NUMERIC_COLS = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti"]

CANONICAL_PURPOSE = [
    "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
    "small_business", "car", "medical", "other",
]
HOME_OWNERSHIP_FIXES = {
    "RENTED": "RENT",
    "MORTGAGED": "MORTGAGE",
}

LOW_THRESHOLD = 2.3
HIGH_THRESHOLD = 2.6

# PL: znane formaty dat do proby przez solver (kolejnosc nieistotna -
#     wszystkie pasujace kandydatury i tak trafiaja do jednej puli)
# EN: known date formats the solver tries (order doesn't matter - all
#     matching candidates go into one pool regardless)
KNOWN_DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d",
    "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
    "%m-%d-%Y", "%m/%d/%Y", "%m.%d.%Y",
    "%b %d, %Y", "%d %b %Y", "%B %d, %Y", "%d %B %Y",
]


# ============================================================
# WARSTWA 1: CZYSZCZENIE DETERMINISTYCZNE (KOD, ZERO LLM)
# LAYER 1: DETERMINISTIC CLEANING (CODE ONLY, ZERO LLM)
# ============================================================

def normalize_purpose(val: str) -> Optional[str]:
    """PL: Dopasowuje wartosc purpose do kanonicznej listy przez fuzzy matching.
    EN: Matches a purpose value to the canonical list via fuzzy matching."""
    if pd.isna(val):
        return np.nan
    cleaned = val.strip().lower().replace(" ", "_").replace("-", "_")
    match, score, _ = process.extractOne(cleaned, CANONICAL_PURPOSE, scorer=fuzz.ratio)
    return match if score >= 80 else np.nan


def normalize_phone(p: str) -> Optional[str]:
    """PL: Wyciaga cyfry i sklada numer telefonu w jeden ustalony format.
    EN: Extracts digits and reassembles the phone number into one fixed format."""
    if pd.isna(p):
        return np.nan
    digits = "".join(ch for ch in str(p) if ch.isdigit())
    if len(digits) != 10:
        return np.nan
    return f"{digits[0:3]}-{digits[3:6]}-{digits[6:10]}"


def run_layer1(input_path: Path) -> pd.DataFrame:
    """PL: Uruchamia cala warstwe 1 - braki danych, separator dziesietny,
    wartosci fizycznie niemozliwe, kategorie, telefon.
    EN: Runs the entire layer 1 - missing values, decimal separator,
    physically impossible values, categories, phone."""
    df = pd.read_csv(input_path, dtype=str)

    df = df.replace(MISSING_MARKERS, np.nan)

    for col in NUMERIC_COLS:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.loc[df["loan_amnt"] < 0, "loan_amnt"] = np.nan
    df["fico_score"] = pd.to_numeric(df["fico_score"], errors="coerce")
    df.loc[(df["fico_score"] < 300) | (df["fico_score"] > 850), "fico_score"] = np.nan
    df.loc[df["dti"] > 100, "dti"] = np.nan
    df.loc[df["annual_inc"] < 0, "annual_inc"] = np.nan

    df["home_ownership"] = df["home_ownership"].str.strip().str.upper()
    df["home_ownership"] = df["home_ownership"].replace(HOME_OWNERSHIP_FIXES)

    df["grade"] = df["grade"].str.strip().str.upper().str.extract(r"([A-G])")

    df["purpose_original"] = df["purpose"]
    df["purpose"] = df["purpose"].apply(normalize_purpose)

    df["phone"] = df["phone"].apply(normalize_phone)

    return df


# ============================================================
# WARSTWA 1.5: DETERMINISTYCZNY SOLVER DAT (KOD, ZERO LLM)
# LAYER 1.5: DETERMINISTIC DATE SOLVER (CODE ONLY, ZERO LLM)
# ============================================================

def get_candidate_dates(raw_str: str) -> list:
    """PL: Probuje sparsowac surowy string data przez wszystkie znane
    formaty, zwraca zbior unikalnych, poprawnych kandydatow.
    EN: Tries parsing a raw date string against all known formats,
    returns a set of unique, valid candidates."""
    if pd.isna(raw_str) or not str(raw_str).strip():
        return []
    s = str(raw_str).strip()
    candidates = set()
    for fmt in KNOWN_DATE_FORMATS:
        try:
            candidates.add(datetime.strptime(s, fmt).date())
        except (ValueError, TypeError):
            continue
    return list(candidates)


def solve_dates(dirty_df: pd.DataFrame) -> pd.DataFrame:
    """PL: Dla kazdego rekordu generuje kandydatow dla application_date i
    issue_date, filtruje kombinacje regula biznesowa (issue_date 1-30 dni
    po application_date). Dokladnie 1 pasujaca kombinacja -> rozstrzygniete
    deterministycznie. 0 lub 2+ -> pozostaje niejednoznaczne (do LLM).
    EN: For each record, generates candidates for application_date and
    issue_date, filters combinations by the business rule (issue_date is
    1-30 days after application_date). Exactly 1 matching combination ->
    resolved deterministically. 0 or 2+ -> stays ambiguous (escalate to LLM)."""
    results = []
    for _, row in dirty_df.iterrows():
        app_candidates = get_candidate_dates(row["application_date"])
        iss_candidates = get_candidate_dates(row["issue_date"])
        valid_pairs = [
            (a, i) for a, i in product(app_candidates, iss_candidates)
            if 1 <= (i - a).days <= 30
        ]
        if len(valid_pairs) == 1:
            app, iss = valid_pairs[0]
            results.append({
                "customer_id": row["customer_id"],
                "solved_application_date": app.strftime("%Y-%m-%d"),
                "solved_issue_date": iss.strftime("%Y-%m-%d"),
                "is_ambiguous": False,
            })
        else:
            results.append({
                "customer_id": row["customer_id"],
                "solved_application_date": None,
                "solved_issue_date": None,
                "is_ambiguous": True,
            })
    return pd.DataFrame(results)


def apply_date_solver(layer2_df: pd.DataFrame, dirty_df: pd.DataFrame) -> pd.DataFrame:
    """PL: Nadpisuje application_date/issue_date w layer2 wynikiem solvera
    tam, gdzie solver jest pewny (99,95% rekordow). Tam gdzie solver nie
    rozstrzyga (naprawde niejednoznaczne), zostawia wartosc ktora juz ma
    layer2 (np. z wczesniejszego czyszczenia LLM) bez zmian.
    EN: Overwrites application_date/issue_date in layer2 with the solver's
    result wherever the solver is confident (99.95% of records). Where the
    solver can't resolve (genuinely ambiguous), leaves whatever value
    layer2 already has (e.g. from earlier LLM cleaning) untouched."""
    solved = solve_dates(dirty_df)
    layer2 = layer2_df.merge(solved, on="customer_id", how="left")

    confident = ~layer2["is_ambiguous"]
    layer2.loc[confident, "application_date"] = layer2.loc[confident, "solved_application_date"]
    layer2.loc[confident, "issue_date"] = layer2.loc[confident, "solved_issue_date"]

    n_confident = confident.sum()
    n_ambiguous = (~confident).sum()
    print(f"Solver dat: {n_confident} rekordow rozstrzygnietych deterministycznie, "
          f"{n_ambiguous} pozostaje niejednoznacznych (bez zmian / do LLM)")

    return layer2.drop(columns=["solved_application_date", "solved_issue_date", "is_ambiguous"])


# ============================================================
# WYKRYWANIE DUPLIKATOW (BLOKOWANIE + FUZZY MATCHING)
# DUPLICATE DETECTION (BLOCKING + FUZZY MATCHING)
# ============================================================

def run_duplicate_detection(input_path: Path) -> pd.DataFrame:
    """PL: Blokowanie (nazwisko+zip) + porownanie pol -> dzieli pary na
    pewne duplikaty, pewne nie-duplikaty i strefe szara (do LLM).
    EN: Blocking (last_name+zip) + field comparison -> splits pairs into
    confident duplicates, confident non-duplicates, and a gray zone (to LLM)."""
    df_full = pd.read_csv(input_path, dtype=str)
    df_full["last_name"] = df_full["full_name"].str.strip().str.split().str[-1].str.upper()

    indexer = recordlinkage.Index()
    indexer.block(["last_name", "zip_code"])
    candidate_pairs = indexer.index(df_full)

    compare = recordlinkage.Compare()
    compare.string("full_name", "full_name", method="jarowinkler", label="name_score")
    compare.string("email", "email", method="jarowinkler", label="email_score")
    compare.exact("phone", "phone", label="phone_score")
    compare.string("street_address", "street_address", method="jarowinkler", label="address_score")
    features = compare.compute(candidate_pairs, df_full)
    features["total_score"] = features[
        ["name_score", "email_score", "phone_score", "address_score"]
    ].sum(axis=1)

    def pair_to_ids(idx_pair):
        return df_full.loc[idx_pair[0], "customer_id"], df_full.loc[idx_pair[1], "customer_id"]

    features["id_a"], features["id_b"] = zip(*[pair_to_ids(p) for p in features.index])

    gray_zone = features[
        (features["total_score"] >= LOW_THRESHOLD) & (features["total_score"] < HIGH_THRESHOLD)
    ]
    gray_zone[["id_a", "id_b", "total_score"]].to_csv(GRAY_ZONE_PATH, index=False)
    print(f"Strefa szara: {len(gray_zone)} par -> {GRAY_ZONE_PATH.name}")

    auto_dup = features[features["total_score"] >= HIGH_THRESHOLD]
    auto_not = features[features["total_score"] < LOW_THRESHOLD]
    auto_dup[["id_a", "id_b", "total_score"]].to_csv(AUTO_DUP_PATH, index=False)
    auto_not[["id_a", "id_b", "total_score"]].to_csv(AUTO_NOT_DUP_PATH, index=False)
    print(f"AUTO-duplikat: {len(auto_dup)}, AUTO-nie-duplikat: {len(auto_not)}")

    return df_full


# ============================================================
# WARSTWA 2: LLM (GEMINI) - TYLKO ESKALACJA
# LAYER 2: LLM (GEMINI) - ESCALATION ONLY
# ============================================================

class RateLimiter:
    def __init__(self, calls_per_minute: int):
        self.min_interval = 60 / calls_per_minute
        self.last_call = 0
        self.lock = asyncio.Lock()

    async def wait(self):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self.last_call = time.monotonic()


rate_limiter = RateLimiter(calls_per_minute=RATE_LIMIT_CALLS_PER_MINUTE)


class CleanedDateNotes(BaseModel):
    customer_id: str
    application_date: Optional[str] = None
    issue_date: Optional[str] = None
    advisor_notes: Optional[str] = None


class CleanedChunk(BaseModel):
    records: list[CleanedDateNotes]


class DuplicateVerdict(BaseModel):
    record_a_id: str
    record_b_id: str
    is_duplicate: bool


class DuplicateVerdicts(BaseModel):
    verdicts: list[DuplicateVerdict]


DATE_NOTES_SCHEMA = """
Jesteś narzędziem do czyszczenia danych bankowych. Otrzymujesz rekordy zawierające
TYLKO: customer_id, application_date, issue_date, advisor_notes.

- daty -> format "YYYY-MM-DD". Wejście może być w różnych formatach (US m/d/y,
  europejski d/m/y, ISO, nazwa miesiąca słownie lub skrótem). issue_date jest
  zawsze 1-30 dni PO application_date - użyj tego do rozstrzygania niejednoznacznych
  przypadków (np. 07/11/2015 - 7 listopada czy 11 lipca?). Data fizycznie
  niemożliwa (np. 31.02) -> null.
- advisor_notes -> popraw literówki, usuń nadmiarowe białe znaki, normalna
  wielkość liter (nie CAPS LOCK), zachowaj oryginalny sens i długość zdania.

Zwróć WSZYSTKIE rekordy z paczki, w tej samej kolejności, customer_id niezmienione.
"""

DUPLICATE_SCHEMA = """
Dla każdej pary rekordów klientów oceń, czy to TEN SAM klient wprowadzony do
systemu dwa razy (duplikat, np. z literówką/inną wielkością liter w danych),
czy DWIE RÓŻNE osoby z podobnymi danymi (to samo nazwisko i zip, ale różny
email/telefon/adres). Zwróć is_duplicate: true/false dla każdej pary.
"""


def chunk_list(data: list, chunk_size: int = CHUNK_SIZE):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


async def clean_chunk(
    chunk: list[dict], semaphore: asyncio.Semaphore, client: genai.Client, chunk_num: int
) -> list[dict]:
    async with semaphore:
        await rate_limiter.wait()
        prompt = DATE_NOTES_SCHEMA + "\n\nOczyść poniższe rekordy:\n" + json.dumps(chunk, default=str)

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config={"response_mime_type": "application/json", "response_schema": CleanedChunk},
                )
                parsed = CleanedChunk.model_validate_json(response.text)

                if len(parsed.records) != len(chunk):
                    raise ValueError(f"dostałem {len(chunk)}, wróciło {len(parsed.records)}")

                print(f"  paczka {chunk_num}: OK ({len(parsed.records)} rekordów)")
                return [r.model_dump() for r in parsed.records]

            except Exception as e:
                wait = 2 ** attempt
                print(f"  paczka {chunk_num}: błąd (próba {attempt + 1}/{MAX_RETRIES}): {e} -> czekam {wait}s")
                await asyncio.sleep(wait)

        print(f"  paczka {chunk_num}: POMINIĘTA po {MAX_RETRIES} próbach")
        return []


async def resolve_gray_zone_duplicates(
    pairs_df: pd.DataFrame, df_full: pd.DataFrame, client: genai.Client
) -> pd.DataFrame:
    fields = ["customer_id", "full_name", "email", "phone", "street_address", "zip_code"]
    payload = []
    for _, row in pairs_df.iterrows():
        rec_a = df_full.loc[df_full["customer_id"] == row["id_a"], fields].iloc[0].to_dict()
        rec_b = df_full.loc[df_full["customer_id"] == row["id_b"], fields].iloc[0].to_dict()
        payload.append({"record_a": rec_a, "record_b": rec_b})

    prompt = DUPLICATE_SCHEMA + "\n\nPary do oceny:\n" + json.dumps(payload, default=str)
    response = await client.aio.models.generate_content(
        model=MODEL_NAME, contents=prompt,
        config={"response_mime_type": "application/json", "response_schema": DuplicateVerdicts},
    )
    parsed = DuplicateVerdicts.model_validate_json(response.text)
    return pd.DataFrame([v.model_dump() for v in parsed.verdicts])


async def run_layer2(df_full: pd.DataFrame) -> None:
    """PL: Pelne uruchomienie warstwy LLM - uzywane TYLKO przy swiezym
    uruchomieniu (SKIP_LLM_RERUN=False), gdy nie ma jeszcze wynikow z
    poprzedniej sesji. Warstwa 1.5 (solver) i tak nadpisze wiekszosc dat
    po tym kroku.
    EN: Full LLM-layer run - used ONLY on a fresh run (SKIP_LLM_RERUN=False)
    when no results from a previous session exist yet. Layer 1.5 (solver)
    will overwrite most dates after this step regardless."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "API_KEY":
        raise RuntimeError(
            "Uzupełnij prawdziwy klucz w GEMINI_API_KEY (sekcja KONFIG). / "
            "Fill in a real key in GEMINI_API_KEY (CONFIG section)."
        )

    df = pd.read_csv(LAYER1_OUTPUT, dtype=str)

    if LAYER2_OUTPUT.exists():
        already_done = pd.read_csv(LAYER2_OUTPUT)
        done_ids = set(already_done["customer_id"])
        print(f"Już oczyszczone wcześniej: {len(done_ids)} rekordów")
    else:
        already_done = pd.DataFrame()
        done_ids = set()

    todo = df[~df["customer_id"].isin(done_ids)]
    records = todo[["customer_id", "application_date", "issue_date", "advisor_notes"]].to_dict(orient="records")
    print(f"Do zrobienia: {len(records)} rekordów ({len(records) // CHUNK_SIZE + 1} paczek)")

    chunks = list(chunk_list(records, CHUNK_SIZE))
    client = genai.Client(api_key=GEMINI_API_KEY)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [clean_chunk(chunk, semaphore, client, i + 1) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)
    new_cleaned = [record for chunk_result in results for record in chunk_result]

    combined = pd.concat([already_done, pd.DataFrame(new_cleaned)], ignore_index=True)
    combined.to_csv(LAYER2_OUTPUT, index=False)
    print(f"Zapisano łącznie: {len(combined)} / {len(df)} rekordów")

    gray = pd.read_csv(GRAY_ZONE_PATH)
    verdicts = await resolve_gray_zone_duplicates(gray, df_full, client)
    verdicts.to_csv(DUPLICATE_VERDICTS_PATH, index=False)
    print(f"Rozstrzygnięto {len(verdicts)} par ze strefy szarej -> {DUPLICATE_VERDICTS_PATH.name}")


# ============================================================
# SCALENIE FINALNE
# FINAL MERGE
# ============================================================

def run_merge() -> pd.DataFrame:
    layer1 = pd.read_csv(LAYER1_OUTPUT, dtype=str)
    layer2 = pd.read_csv(LAYER2_OUTPUT, dtype=str)
    auto_dup = pd.read_csv(AUTO_DUP_PATH, dtype=str)
    llm_verdicts = pd.read_csv(DUPLICATE_VERDICTS_PATH, dtype=str)

    llm_cols = ["application_date", "issue_date", "advisor_notes"]
    final = layer1.drop(columns=llm_cols).merge(layer2, on="customer_id", how="left")

    final["is_duplicate"] = False
    final["duplicate_of"] = None

    for _, row in auto_dup.iterrows():
        final.loc[final["customer_id"] == row["id_a"], "is_duplicate"] = True
        final.loc[final["customer_id"] == row["id_a"], "duplicate_of"] = row["id_b"]

    llm_verdicts["is_duplicate"] = llm_verdicts["is_duplicate"].astype(str).str.lower() == "true"
    confirmed_by_llm = llm_verdicts[llm_verdicts["is_duplicate"]]
    for _, row in confirmed_by_llm.iterrows():
        final.loc[final["customer_id"] == row["record_a_id"], "is_duplicate"] = True
        final.loc[final["customer_id"] == row["record_a_id"], "duplicate_of"] = row["record_b_id"]

    final.to_csv(FINAL_OUTPUT, index=False)
    print(f"\nZapisano finalny plik: {FINAL_OUTPUT.name}")
    print(f"Wiersze: {len(final)}, kolumny: {len(final.columns)}")
    print(f"Oznaczonych jako duplikat: {final['is_duplicate'].sum()}")

    return final


# ============================================================
# GŁÓWNA ORKIESTRACJA
# MAIN ORCHESTRATION
# ============================================================

def main() -> None:
    print("=== KROK 1/5: Warstwa deterministyczna (kod) ===")
    layer1_df = run_layer1(DIRTY_DATA_PATH)
    layer1_df.to_csv(LAYER1_OUTPUT, index=False)
    print(f"Zapisano: {LAYER1_OUTPUT.name} ({len(layer1_df)} wierszy)\n")

    print("=== KROK 2/5: Wykrywanie duplikatów (blokowanie + fuzzy matching) ===")
    df_full = run_duplicate_detection(DIRTY_DATA_PATH)
    print()

    if SKIP_LLM_RERUN:
        print("=== KROK 3/5: POMINIĘTY (SKIP_LLM_RERUN=True) - używam istniejących wyników LLM ===")
        if not LAYER2_OUTPUT.exists() or not DUPLICATE_VERDICTS_PATH.exists():
            raise RuntimeError(
                f"SKIP_LLM_RERUN=True, ale brakuje {LAYER2_OUTPUT.name} lub "
                f"{DUPLICATE_VERDICTS_PATH.name}. Ustaw SKIP_LLM_RERUN=False na świeży bieg. / "
                f"SKIP_LLM_RERUN=True, but {LAYER2_OUTPUT.name} or {DUPLICATE_VERDICTS_PATH.name} "
                f"is missing. Set SKIP_LLM_RERUN=False for a fresh run."
            )
        print()
    else:
        print("=== KROK 3/5: Warstwa LLM (Gemini) - daty, notatki, strefa szara ===")
        asyncio.run(run_layer2(df_full))
        print()

    print("=== KROK 4/5: Solver dat (deterministyczny, zero LLM) ===")
    dirty_df = pd.read_csv(DIRTY_DATA_PATH, dtype=str)
    layer2_updated = apply_date_solver(pd.read_csv(LAYER2_OUTPUT, dtype=str), dirty_df)
    layer2_updated.to_csv(LAYER2_OUTPUT, index=False)
    print(f"Zaktualizowano: {LAYER2_OUTPUT.name}\n")

    print("=== KROK 5/5: Scalenie finalne ===")
    run_merge()


if __name__ == "__main__":
    main()
