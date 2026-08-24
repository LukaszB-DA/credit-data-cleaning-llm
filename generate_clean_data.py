"""
generate_clean_data.py
Generuje CZYSTĄ bazę referencyjną (ground truth) - 100k rekordów kredytowych,
statystycznie kalibrowaną na realnych rozkładach Lending Club.
Ta baza nigdy nie trafia do LLM - służy jako klucz odpowiedzi do pomiaru
skuteczności skryptu czyszczącego.
"""

import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

N = 100_000
SEED = 42
rng = np.random.default_rng(SEED)
fake = Faker("en_US")
Faker.seed(SEED)

# ============================================================
# 1. DANE FINANSOWE (wzajemnie spójne)
# ============================================================

# loan_amnt: rozkład lognormalny, przycięty do realnego zakresu LC, zaokrąglony do 50
raw_amounts = rng.lognormal(mean=9.5, sigma=0.45, size=N)
loan_amnt = np.clip(raw_amounts, 1000, 40000)
loan_amnt = np.round(loan_amnt / 50) * 50

# grade: ocena ryzyka, rozkład przesunięty w stronę B/C
grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
grade_probs = [0.16, 0.29, 0.28, 0.15, 0.07, 0.03, 0.02]
grade = rng.choice(grades, size=N, p=grade_probs)

# int_rate: zależny od grade + szum
base_rate = {'A': 7.0, 'B': 10.5, 'C': 13.5, 'D': 17.0, 'E': 21.0, 'F': 25.0, 'G': 28.5}
base = pd.Series(grade).map(base_rate).to_numpy()
noise = rng.normal(loc=0, scale=1.2, size=N)
int_rate = np.clip(base + noise, 5.31, 30.99)
int_rate = np.round(int_rate, 2)

# term: zależny od kwoty pożyczki
prob_60m = np.clip((loan_amnt - 5000) / 35000, 0.1, 0.75)
term = np.where(rng.random(N) < prob_60m, 60, 36)

# installment: prawdziwy wzór annuitetowy (spójność matematyczna, brak losowości)
r = (int_rate / 100) / 12
n = term
installment = loan_amnt * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
installment = np.round(installment, 2)

# annual_inc: prawoskośny rozkład dochodów
annual_inc = rng.lognormal(mean=11.0, sigma=0.5, size=N)
annual_inc = np.clip(annual_inc, 15000, 400000)
annual_inc = np.round(annual_inc, 2)

# dti: zależny od dochodu i raty (logika: im wyższa rata względem dochodu, tym wyższe DTI)
monthly_inc = annual_inc / 12
other_debt_ratio = rng.uniform(0.05, 0.30, size=N)  # inne zobowiązania klienta
dti = ((installment + monthly_inc * other_debt_ratio) / monthly_inc) * 100
dti = np.clip(dti, 0, 45)
dti = np.round(dti, 2)

# fico: skorelowane z grade (lepsza ocena -> wyższy fico)
base_fico = {'A': 780, 'B': 730, 'C': 690, 'D': 665, 'E': 645, 'F': 630, 'G': 620}
fico_base = pd.Series(grade).map(base_fico).to_numpy()
fico_noise = rng.normal(0, 12, size=N)
fico_score = np.clip(fico_base + fico_noise, 600, 850).astype(int)

# ============================================================
# 2. DANE KATEGORYCZNE
# ============================================================

home_ownership = rng.choice(
    ['RENT', 'MORTGAGE', 'OWN', 'OTHER'], size=N, p=[0.42, 0.44, 0.11, 0.03]
)

verification_status = rng.choice(
    ['Verified', 'Source Verified', 'Not Verified'], size=N, p=[0.30, 0.35, 0.35]
)

purpose = rng.choice(
    ['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase',
     'small_business', 'car', 'medical', 'other'],
    size=N, p=[0.55, 0.20, 0.07, 0.05, 0.04, 0.03, 0.03, 0.03]
)

emp_length_options = ['< 1 year', '1 year', '2 years', '3 years', '4 years',
                       '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
emp_length_probs = [0.10, 0.09, 0.10, 0.09, 0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.20]
emp_length = rng.choice(emp_length_options, size=N, p=emp_length_probs)

loan_status = rng.choice(
    ['Fully Paid', 'Current', 'Charged Off', 'Late (31-120 days)'],
    size=N, p=[0.45, 0.35, 0.13, 0.07]
)

# ============================================================
# 3. DANE OSOBOWE / TEKSTOWE (pole popisowe pod LLM)
# ============================================================

first_names = [fake.first_name() for _ in range(N)]
last_names = [fake.last_name() for _ in range(N)]
full_name = [f"{f} {l}" for f, l in zip(first_names, last_names)]
email = [f"{f}.{l}{rng.integers(1,999)}@{fake.free_email_domain()}".lower()
         for f, l in zip(first_names, last_names)]
phone = [fake.numerify("###-###-####") for _ in range(N)]

street_address = [fake.street_address() for _ in range(N)]
city = [fake.city() for _ in range(N)]
state = [fake.state_abbr() for _ in range(N)]
zip_code = [fake.zipcode() for _ in range(N)]

emp_title = [fake.job() for _ in range(N)]

# ============================================================
# 4. DATY
# ============================================================

start_date = datetime(2015, 1, 1)
issue_date = [start_date + timedelta(days=int(d)) for d in rng.integers(0, 365 * 9, size=N)]
# data aplikacji zawsze nieco wcześniej niż data wydania pożyczki
application_date = [d - timedelta(days=int(x)) for d, x in zip(issue_date, rng.integers(1, 30, size=N))]

# ============================================================
# 5. NOTATKA DORADCY (wolny tekst - kluczowe pole pod LLM cleaning)
# ============================================================

note_templates = [
    "Stable income, employed {emp_len} at current job, verified via pay stubs.",
    "Self-employed, income verified through bank statements, {emp_len} in business.",
    "Applicant has {emp_len} employment history, no red flags in credit review.",
    "Co-signer available, primary applicant employed {emp_len}.",
    "Recently changed jobs, {emp_len} at new employer, income consistent with prior role.",
    "Long-term customer, {emp_len} employment, strong repayment history on prior loans.",
]
advisor_notes = [
    rng.choice(note_templates).replace("{emp_len}", str(el))
    for el in emp_length
]

# ============================================================
# 6. SKŁADANIE DATAFRAME
# ============================================================

df = pd.DataFrame({
    "customer_id": [f"LC{100000 + i}" for i in range(N)],
    "full_name": full_name,
    "email": email,
    "phone": phone,
    "street_address": street_address,
    "city": city,
    "state": state,
    "zip_code": zip_code,
    "loan_amnt": loan_amnt,
    "term_months": term,
    "int_rate": int_rate,
    "installment": installment,
    "grade": grade,
    "emp_title": emp_title,
    "emp_length": emp_length,
    "home_ownership": home_ownership,
    "annual_inc": annual_inc,
    "verification_status": verification_status,
    "purpose": purpose,
    "dti": dti,
    "fico_score": fico_score,
    "loan_status": loan_status,
    "application_date": application_date,
    "issue_date": issue_date,
    "advisor_notes": advisor_notes,
})

df.to_csv("/home/claude/clean_data.csv", index=False)
print(f"OK - zapisano {len(df)} wierszy, {len(df.columns)} kolumn")
print(df.head(3).T)
