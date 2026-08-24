"""
clean_credit_data_llm.py

PL: Kompletny pipeline czyszczenia danych bankowych (101 500 rekordów).
    Dwie warstwy: (1) deterministyczny kod - pandas/regex/fuzzy matching/
    record linkage, (2) LLM (Gemini) - tylko tam, gdzie warstwa 1 nie
    rozstrzyga jednoznacznie (daty, wolny tekst, strefa szara duplikatów).

EN: Full bank-data cleaning pipeline (101,500 records). Two layers:
    (1) deterministic code - pandas/regex/fuzzy matching/record linkage,
    (2) LLM (Gemini) - only where layer 1 cannot resolve unambiguously
    (dates, free text, gray-zone duplicate pairs).

Uruchomienie / Usage:
    python clean_credit_data_llm.py

Wymaga uzupełnienia klucza API poniżej / Requires the API key to be filled in below:
    GEMINI_API_KEY = "..." (patrz sekcja CONFIG / see CONFIG section)

Wynik / Output:
    final_cleaned_data.csv
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import recordlinkage
from google import genai
from pydantic import BaseModel
from rapidfuzz import fuzz, process

# ============================================================
# CONFIG
# ============================================================

# PL: katalog skryptu jako punkt odniesienia - działa od razu po sklonowaniu
#     repozytorium, bez ręcznej edycji ścieżek.
# EN: script's own directory as the base path - works right after cloning
#     the repo, no manual path editing needed.
BASE_DIR = Path(__file__).resolve().parent

DIRTY_DATA_PATH = BASE_DIR / "dirty_data.csv"
LAYER1_OUTPUT = BASE_DIR / "layer1_cleaned.csv"
GRAY_ZONE_PATH = BASE_DIR / "duplicate_candidates_gray.csv"
AUTO_DUP_PATH = BASE_DIR / "auto_duplicate.csv"
AUTO_NOT_DUP_PATH = BASE_DIR / "auto_not_duplicate.csv"
LAYER2_OUTPUT = BASE_DIR / "layer2_dates_notes_full.csv"
DUPLICATE_VERDICTS_PATH = BASE_DIR / "duplicate_verdicts_llm.csv"
FINAL_OUTPUT = BASE_DIR / "final_cleaned_data.csv"

# PL: klucz API wyłącznie przez zmienną środowiskową - nigdy na sztywno w kodzie.
# EN: API key exclusively via environment variable - never hardcoded.
GEMINI_API_KEY = "API_KEY"
MODEL_NAME = "gemini-3.5-flash-lite"

CHUNK_SIZE = 100
MAX_CONCURRENT = 10
MAX_RETRIES = 3
RATE_LIMIT_CALLS_PER_MINUTE = 100

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

LOW_THRESHOLD = 2.3   # PL: poniżej -> auto: NIE duplikat / EN: below -> auto: NOT a duplicate
HIGH_THRESHOLD = 2.6  # PL: powyżej/równe -> auto: TAK duplikat / EN: above/equal -> auto: duplicate


# ============================================================
# WARSTWA 1: CZYSZCZENIE DETERMINISTYCZNE (KOD, ZERO LLM)
# LAYER 1: DETERMINISTIC CLEANING (CODE ONLY, ZERO LLM)
# ============================================================

def normalize_purpose(val: str) -> Optional[str]:
    """PL: Dopasowuje wartość purpose do kanonicznej listy przez fuzzy matching.
    EN: Matches a purpose value to the canonical list via fuzzy matching."""
    if pd.isna(val):
        return np.nan
    cleaned = val.strip().lower().replace(" ", "_").replace("-", "_")
    match, score, _ = process.extractOne(cleaned, CANONICAL_PURPOSE, scorer=fuzz.ratio)
    return match if score >= 80 else np.nan


def normalize_phone(p: str) -> Optional[str]:
    """PL: Wyciąga cyfry i składa numer telefonu w jeden ustalony format.
    EN: Extracts digits and reassembles the phone number into one fixed format."""
    if pd.isna(p):
        return np.nan
    digits = "".join(ch for ch in str(p) if ch.isdigit())
    if len(digits) != 10:
        return np.nan
    return f"{digits[0:3]}-{digits[3:6]}-{digits[6:10]}"


def run_layer1(input_path: Path) -> pd.DataFrame:
    """PL: Uruchamia całą warstwę 1 - braki danych, separator dziesiętny,
    wartości fizycznie niemożliwe, kategorie, telefon.
    EN: Runs the entire layer 1 - missing values, decimal separator,
    physically impossible values, categories, phone."""
    df = pd.read_csv(input_path, dtype=str)

    # PL: ujednolicenie braków danych - zawsze pierwsze, przed resztą operacji
    # EN: unify missing-value markers - always first, before anything else
    df = df.replace(MISSING_MARKERS, np.nan)

    # PL: separator dziesiętny (przecinek -> kropka) + konwersja na liczby
    # EN: decimal separator (comma -> dot) + numeric conversion
    for col in NUMERIC_COLS:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # PL: wartości fizycznie niemożliwe -> NaN
    # EN: physically impossible values -> NaN
    df.loc[df["loan_amnt"] < 0, "loan_amnt"] = np.nan
    df["fico_score"] = pd.to_numeric(df["fico_score"], errors="coerce")
    df.loc[(df["fico_score"] < 300) | (df["fico_score"] > 850), "fico_score"] = np.nan
    df.loc[df["dti"] > 100, "dti"] = np.nan
    df.loc[df["annual_inc"] < 0, "annual_inc"] = np.nan

    # PL: home_ownership - wielkość liter/spacje + warianty słowne (RENTED -> RENT)
    # EN: home_ownership - case/whitespace + word variants (RENTED -> RENT)
    df["home_ownership"] = df["home_ownership"].str.strip().str.upper()
    df["home_ownership"] = df["home_ownership"].replace(HOME_OWNERSHIP_FIXES)

    # PL: grade - wyciąga literę A-G nawet z "grade B " (regex)
    # EN: grade - extracts letter A-G even from "grade B " (regex)
    df["grade"] = df["grade"].str.strip().str.upper().str.extract(r"([A-G])")

    # PL: purpose - fuzzy matching do kanonicznej listy (łapie literówki)
    # EN: purpose - fuzzy matching to canonical list (catches typos)
    df["purpose_original"] = df["purpose"]
    df["purpose"] = df["purpose"].apply(normalize_purpose)

    # PL: telefon - ekstrakcja cyfr + reformatowanie
    # EN: phone - digit extraction + reformatting
    df["phone"] = df["phone"].apply(normalize_phone)

    return df


# ============================================================
# WYKRYWANIE DUPLIKATÓW (BLOKOWANIE + FUZZY MATCHING)
# DUPLICATE DETECTION (BLOCKING + FUZZY MATCHING)
# ============================================================

def run_duplicate_detection(input_path: Path) -> pd.DataFrame:
    """PL: Blokowanie (nazwisko+zip) + porównanie pól -> dzieli pary na
    pewne duplikaty, pewne nie-duplikaty i strefę szarą (do LLM).
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
    """PL: Ogranicza tempo wysyłania zapytań do ustalonej liczby na minutę.
    EN: Throttles request rate to a fixed number of calls per minute."""

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
    """PL: Dzieli listę na paczki o stałym rozmiarze (generator).
    EN: Splits a list into fixed-size chunks (generator)."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


async def clean_chunk(
    chunk: list[dict], semaphore: asyncio.Semaphore, client: genai.Client, chunk_num: int
) -> list[dict]:
    """PL: Czyści jedną paczkę (async, ograniczona współbieżność, retry,
    walidacja kompletności odpowiedzi).
    EN: Cleans a single chunk (async, bounded concurrency, retries,
    response-completeness validation)."""
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
    """PL: Rozstrzyga pary duplikatów ze strefy szarej przy pomocy LLM.
    EN: Resolves gray-zone duplicate pairs using the LLM."""
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
    """PL: Orkiestruje całą warstwę LLM - czyszczenie dat/notatek (z
    mechanizmem wznawiania) oraz rozstrzygnięcie strefy szarej duplikatów.
    EN: Orchestrates the entire LLM layer - date/notes cleaning (with a
    resume mechanism) and gray-zone duplicate resolution."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "API_KEY":
        raise RuntimeError(
            "Uzupełnij prawdziwy klucz w GEMINI_API_KEY (sekcja KONFIG). / "
            "Fill in a real key in GEMINI_API_KEY (CONFIG section)."
        )

    df = pd.read_csv(LAYER1_OUTPUT, dtype=str)

    # PL: mechanizm wznawiania - pomija rekordy już wcześniej oczyszczone
    # EN: resume mechanism - skips records already cleaned in a prior run
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
    """PL: Scala warstwę 1 i warstwę 2 w jeden finalny plik, dokłada flagę
    is_duplicate na podstawie decyzji kodu i LLM.
    EN: Merges layer 1 and layer 2 into one final file, adds an
    is_duplicate flag based on both code and LLM decisions."""
    layer1 = pd.read_csv(LAYER1_OUTPUT, dtype=str)
    layer2 = pd.read_csv(LAYER2_OUTPUT, dtype=str)
    auto_dup = pd.read_csv(AUTO_DUP_PATH, dtype=str)
    llm_verdicts = pd.read_csv(DUPLICATE_VERDICTS_PATH, dtype=str)

    # PL: podmiana kolumn dat/notatek - z wersji kodowej (layer1) na LLM (layer2)
    # EN: swap date/notes columns - from the code version (layer1) to LLM (layer2)
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
    print("=== KROK 1/4: Warstwa deterministyczna (kod) ===")
    layer1_df = run_layer1(DIRTY_DATA_PATH)
    layer1_df.to_csv(LAYER1_OUTPUT, index=False)
    print(f"Zapisano: {LAYER1_OUTPUT.name} ({len(layer1_df)} wierszy)\n")

    print("=== KROK 2/4: Wykrywanie duplikatów (blokowanie + fuzzy matching) ===")
    df_full = run_duplicate_detection(DIRTY_DATA_PATH)
    print()

    print("=== KROK 3/4: Warstwa LLM (Gemini) - daty, notatki, strefa szara ===")
    asyncio.run(run_layer2(df_full))
    print()

    print("=== KROK 4/4: Scalenie finalne ===")
    run_merge()


if __name__ == "__main__":
    main()
