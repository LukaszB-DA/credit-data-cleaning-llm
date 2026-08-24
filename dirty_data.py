"""
dirty_data.py
Nakłada KONTROLOWANY brud na clean_data.csv, tworząc dirty_data.csv.
Każdy typ zabrudzenia jest logowany, dzięki czemu można później dokładnie
zmierzyć skuteczność narzędzia czyszczącego (ground truth = clean_data.csv).
"""

import numpy as np
import pandas as pd
import random

SEED = 7
rng = np.random.default_rng(SEED)
random.seed(SEED)

df = pd.read_csv("/home/claude/clean_data.csv", dtype=str)  # wczytujemy jako string - sami decydujemy o "brudzie"
N = len(df)

log = []  # log każdej zmiany: (customer_id, kolumna, typ_brudu)


def dirty_sample(col, frac, marker_pool, weights=None):
    """Losowo podmienia frac% wartości w kolumnie na jeden z markerów braku danych."""
    idx = rng.choice(N, size=int(N * frac), replace=False)
    markers = rng.choice(marker_pool, size=len(idx), p=weights)
    for i, m in zip(idx, markers):
        log.append((df.at[i, "customer_id"], col, "missing_marker"))
        df.at[i, col] = m


# ============================================================
# 1. BRAKI DANYCH - różne reprezentacje w różnych kolumnach
# ============================================================

dirty_sample("annual_inc", 0.04, ["", "NULL", "N/A", "unknown", "-1"])
dirty_sample("emp_title", 0.06, ["", "n/a", "N/A", "unknown", "???"])
dirty_sample("dti", 0.03, ["", "NULL", "nan", "999"])
dirty_sample("zip_code", 0.02, ["", "00000", "N/A"])
dirty_sample("phone", 0.03, ["", "N/A", "no phone"])

# ============================================================
# 2. NIESPÓJNE FORMATY DAT
# ============================================================

def messy_date(date_str):
    d = pd.to_datetime(date_str)
    fmt = rng.choice(["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%d %b %Y", "%Y/%m/%d"])
    return d.strftime(fmt)

for col in ["application_date", "issue_date"]:
    idx = rng.choice(N, size=int(N * 0.5), replace=False)  # połowa dat w "obcym" formacie
    for i in idx:
        old = df.at[i, col]
        df.at[i, col] = messy_date(old)
        log.append((df.at[i, "customer_id"], col, "date_format"))

# ============================================================
# 3. SEPARATOR DZIESIĘTNY (przecinek zamiast kropki) - część rekordów
# ============================================================

for col in ["loan_amnt", "int_rate", "dti", "annual_inc"]:
    idx = rng.choice(N, size=int(N * 0.08), replace=False)
    for i in idx:
        val = df.at[i, col]
        if val not in ("", "NULL", "N/A", "unknown", "-1", "nan", "999"):
            df.at[i, col] = str(val).replace(".", ",")
            log.append((df.at[i, "customer_id"], col, "decimal_comma"))

# ============================================================
# 4. NIESPÓJNE KATEGORIE (wielkość liter, spacje, warianty zapisu)
# ============================================================

home_ownership_variants = {
    "RENT": ["rent", "Rent", " RENT", "RENT ", "rented"],
    "MORTGAGE": ["mortgage", "Mortgage", "MORTGAGE ", "mortgaged"],
    "OWN": ["own", "Own", " OWN"],
    "OTHER": ["other", "Other", "OTHER "],
}
idx = rng.choice(N, size=int(N * 0.15), replace=False)
for i in idx:
    orig = df.at[i, "home_ownership"]
    variants = home_ownership_variants.get(orig)
    if variants:
        df.at[i, "home_ownership"] = rng.choice(variants)
        log.append((df.at[i, "customer_id"], "home_ownership", "category_variant"))

grade_idx = rng.choice(N, size=int(N * 0.10), replace=False)
for i in grade_idx:
    g = df.at[i, "grade"]
    df.at[i, "grade"] = rng.choice([g.lower(), f" {g}", f"{g} ", f"grade {g}"])
    log.append((df.at[i, "customer_id"], "grade", "category_variant"))

purpose_typo_map = {
    "debt_consolidation": ["Debt Consolidation", "debt consolidation", "DEBT_CONSOLIDATION", "dept_consolidation"],
    "credit_card": ["Credit Card", "creditcard", "credit-card"],
    "home_improvement": ["Home Improvement", "home_improvment", "homeimprovement"],
    "small_business": ["Small Business", "small buisness"],
}
idx = rng.choice(N, size=int(N * 0.10), replace=False)
for i in idx:
    orig = df.at[i, "purpose"]
    variants = purpose_typo_map.get(orig)
    if variants:
        df.at[i, "purpose"] = rng.choice(variants)
        log.append((df.at[i, "customer_id"], "purpose", "category_variant"))

# ============================================================
# 5. SZUM W TEKŚCIE (notatki doradcy - literówki, skróty, wielkie litery)
# ============================================================

typo_swaps = [("employed", "emplyed"), ("verified", "verifed"), ("business", "buisness"),
              ("income", "incme"), ("history", "histroy"), ("consistent", "consistant")]

idx = rng.choice(N, size=int(N * 0.20), replace=False)
for i in idx:
    text = df.at[i, "advisor_notes"]
    for a, b in typo_swaps:
        if a in text and rng.random() < 0.5:
            text = text.replace(a, b)
    if rng.random() < 0.3:
        text = text.upper()
    elif rng.random() < 0.3:
        text = "  " + text + "   "
    df.at[i, "advisor_notes"] = text
    log.append((df.at[i, "customer_id"], "advisor_notes", "text_noise"))

# ============================================================
# 6. WARTOŚCI ODSTAJĄCE / NIELOGICZNE
# ============================================================

idx = rng.choice(N, size=int(N * 0.01), replace=False)
for i in idx:
    df.at[i, "loan_amnt"] = str(-abs(rng.integers(1000, 40000)))
    log.append((df.at[i, "customer_id"], "loan_amnt", "outlier_negative"))

idx = rng.choice(N, size=int(N * 0.01), replace=False)
for i in idx:
    df.at[i, "fico_score"] = "9999"
    log.append((df.at[i, "customer_id"], "fico_score", "outlier_impossible"))

idx = rng.choice(N, size=int(N * 0.01), replace=False)
for i in idx:
    df.at[i, "dti"] = str(rng.integers(150, 999))
    log.append((df.at[i, "customer_id"], "dti", "outlier_impossible"))

# ============================================================
# 7. NIESPÓJNE FORMATY TELEFONU
# ============================================================

def messy_phone(p):
    digits = "".join(ch for ch in p if ch.isdigit())
    if len(digits) != 10:
        return p
    fmt = rng.choice(["({}) {}-{}", "{}.{}.{}", "{}-{}-{}", "{}{}{}"])
    return fmt.format(digits[:3], digits[3:6], digits[6:])

idx = rng.choice(N, size=int(N * 0.30), replace=False)
for i in idx:
    old = df.at[i, "phone"]
    if old not in ("", "N/A", "no phone"):
        df.at[i, "phone"] = messy_phone(old)
        log.append((df.at[i, "customer_id"], "phone", "phone_format"))

# ============================================================
# 8. DUPLIKATY KLIENTÓW (z wariacjami - typowe w realnych bazach)
# ============================================================

dup_idx = rng.choice(N, size=int(N * 0.015), replace=False)
duplicates = df.loc[dup_idx].copy()
duplicates["customer_id"] = [f"DUP{1000+i}" for i in range(len(duplicates))]

# lekkie wariacje w duplikatach - tak jak przy błędzie ponownego wprowadzenia klienta
for i in duplicates.index:
    if rng.random() < 0.5:
        duplicates.at[i, "full_name"] = duplicates.at[i, "full_name"].upper()
    if rng.random() < 0.3:
        duplicates.at[i, "email"] = duplicates.at[i, "email"].replace("@", "+dup@")

dup_log = pd.DataFrame({
    "duplicate_customer_id": duplicates["customer_id"].values,
    "original_customer_id": df.loc[dup_idx, "customer_id"].values,
})

df_final = pd.concat([df, duplicates], ignore_index=True)
df_final = df_final.sample(frac=1, random_state=SEED).reset_index(drop=True)  # tasujemy wiersze

# ============================================================
# ZAPIS
# ============================================================

df_final.to_csv("/home/claude/dirty_data.csv", index=False)
dup_log.to_csv("/home/claude/duplicate_ground_truth.csv", index=False)
pd.DataFrame(log, columns=["customer_id", "column", "dirt_type"]).to_csv(
    "/home/claude/dirt_log.csv", index=False
)

print(f"dirty_data.csv: {len(df_final)} wierszy (w tym {len(duplicates)} duplikatów)")
print(f"dirt_log.csv: {len(log)} zalogowanych zmian")
print(df_final["dirt_type"].unique() if "dirt_type" in df_final.columns else "log zapisany osobno")
