# ETL z LLM — inteligentne czyszczenie danych bankowych na dużą skalę

🇵🇱 Polski | 🇬🇧 [English](README.md)

## O projekcie

Drugi etap mojego portfolio z zakresu czyszczenia danych — po ręcznym, klasycznym
czyszczeniu danych kredytowych (Excel/Power Query + Python), ten projekt pokazuje
**świadome** wykorzystanie LLM w pipeline ETL: nie "wrzucam wszystko do modelu bo
się da", tylko architekturę, która używa taniego, deterministycznego kodu tam,
gdzie to wystarcza, a LLM tylko tam, gdzie faktycznie daje przewagę.

**Cel:** oczyścić 101 500 rekordów bankowych (kredyty, styl Lending Club) w jednym,
zautomatyzowanym przebiegu, łącząc klasyczne techniki (pandas, regex, fuzzy
matching, record linkage, deterministyczny solver kombinatoryczny) z modelem
językowym (Gemini) tam, gdzie reguły faktycznie zawodzą.

> **Uwaga językowa:** prompty wysyłane do modelu oraz komunikaty w `print()`
> są napisane po polsku — to był świadomy wybór na etapie budowy, nie
> przeoczenie. Komentarze w kodzie są sukcesywnie uzupełniane wersją
> angielską, docelowo repo będzie miało dwujęzyczne README (PL/EN).

## Dane

Ponieważ realne dane bankowe (np. pełny zbiór Lending Club) nie były dostępne
w wystarczająco "surowej" formie (dostępne kopie były już wstępnie zakodowane/
oczyszczone, bez pól tekstowych), przygotowałem:

- **`clean_data.csv`** — 100 000 syntetycznych rekordów, statystycznie
  kalibrowanych na realnych rozkładach Lending Club (kwoty, oprocentowanie
  skorelowane z oceną ryzyka, wzory finansowe na ratę itd.) — to jest ground
  truth, nigdy nie widziany przez model.
- **`dirty_data.csv`** — 101 500 wierszy (100k + 1500 zduplikowanych klientów
  z wariacjami), z kontrolowanym, w pełni zalogowanym "brudem": braki danych
  w 5 różnych zapisach, 5 formatów dat, separator dziesiętny przecinek/kropka,
  warianty kategorii, literówki w tekście, wartości fizycznie niemożliwe.

Dzięki własnoręcznie wygenerowanemu brudowi mam dokładny ground truth — mogę
precyzyjnie zmierzyć skuteczność każdego etapu czyszczenia, zamiast tylko
zakładać że "wygląda dobrze".

**Eksploracja danych:** zanim opracowałem reguły czyszczące, przeglądałem dane
iteracyjnie w Data Wrangler (VS Code/Jupyter) — filtry, `groupby`, sortowanie
ASC/DESC — do szybkiego wyłapywania outlierów, wartości nielogicznych i
wzorców w brudnych kolumnach, zanim to przełożyło się na konkretne reguły w
kodzie (np. `pd.to_numeric(errors="coerce")` do wykrywania niesparsowalnych
wartości, regex przy ekstrakcji `grade`).

## Architektura — trzy warstwy

### Warstwa 1: kod deterministyczny (zero LLM)

Obsługuje wszystko, co da się rozwiązać regułami/dopasowaniem tekstu:

| Problem | Metoda | Skuteczność |
|---|---|---|
| Braki danych (5 zapisów) | `.replace()` na markery + `pd.to_numeric(errors="coerce")` do wykrywania | 100% |
| Separator dziesiętny | `pd.to_numeric(errors="coerce")` | 100% |
| Wartości fizycznie niemożliwe | reguły zakresowe | 100% |
| `home_ownership`, `purpose` (literówki) | fuzzy matching (rapidfuzz) | 100% |
| Telefon | ekstrakcja cyfr + reformatowanie | 100% |
| Duplikaty klientów | blokowanie + porównanie (recordlinkage) | 99,3% par ze 100% trafnością |

### Warstwa 1.5: deterministyczny solver dat (zero LLM) — odkryty w toku pracy

To jest najważniejsza rewizja tego projektu, więc opiszę **jak do niej doszedłem**,
nie tylko efekt końcowy — bo sam proces jest wartościowszy niż wynik.

Pierwotnie daty (`application_date`, `issue_date`) czyścił wyłącznie LLM —
zmierzona skuteczność 98,78%/99,52%. Zastosowałem **samowalidujący check**, który
nie wymaga znajomości prawdziwych wartości: sprawdza, czy `issue_date` mieści
się w regule biznesowej (1-30 dni po `application_date`), którą **sam model**
dostał w promptcie. Rekordy łamiące tę regułę są z definicji podejrzane — bez
oglądania ground truth. Wyłapało to **1493 rekordy** na 101 500.

Dociekając dalej: te "brudne" daty (US m/d/y, EU d/m/y, ISO, nazwa miesiąca) mają
w rzeczywistości **bardzo mały, policzalny zbiór możliwych interpretacji** — max
12 znanych formatów × 12 formatów dla dwóch pól. Zastosowałem solver: generuje
wszystkich kandydatów dla obu dat, filtruje kombinacje regułą "issue_date jest
1-30 dni po application_date", i:

- **dokładnie 1 pasująca kombinacja** → rozstrzygnięte deterministycznie, zero LLM
- **0 lub 2+ kombinacji** → naprawdę niejednoznaczne, wymaga dodatkowego sygnału

Wynik na całym zbiorze: **101 452 z 101 500 rekordów (99,95%) rozstrzygniętych
w pełni deterministycznie**, zweryfikowane względem ground truth na **100,000%
zgodności** tam, gdzie solver twierdził że jest pewny. Tylko **48 rekordów**
(0,05%) pozostaje naprawdę niejednoznacznych.

**Wniosek, który to podważyło:** LLM w ogóle nie był potrzebny do rozwiązania
tego konkretnego problemu w tej skali. Zadanie miało strukturę zamkniętego
problemu kombinatorycznego z jawną regułą rozstrzygającą — dokładnie tam, gdzie
klasyczny solver zawsze wygra z modelem probabilistycznym, bo stosuje regułę
ze 100% konsekwencją, a LLM (nawet mając dokładnie tę samą regułę w promptcie)
— nie zawsze.

### Warstwa 2: LLM (Gemini 3.5 Flash-Lite) — tylko prawdziwa eskalacja

Po wprowadzeniu solvera, LLM zostaje potrzebny wyłącznie tam, gdzie problem
**nie ma** zamkniętej, wyliczalnej przestrzeni rozwiązań:

- **`advisor_notes`** — wolny tekst, nieskończona liczba wariantów literówek,
  nie da się tego wyczerpać regułami.
- **~48 naprawdę niejednoznacznych dat** (0,05% rekordów) — solver nie ma jak
  rozstrzygnąć, bo informacja fizycznie nie istnieje w samych danych.
- **Strefa szara duplikatów** — ~0,7% par, gdzie kod nie jest pewny.

## Wyniki końcowe (101 500 rekordów)

| Metryka | Wynik |
|---|---|
| `application_date` | 99,999% |
| `issue_date` | 100,000% |
| Duplikaty (kod) | 1466 par oznaczonych jako duplikat, 100% trafność |
| Duplikaty (LLM, strefa szara) | 17/17 poprawnych ocen (7 potwierdzonych duplikatów + 10 poprawnie odrzuconych) |
| Duplikaty łącznie | 1473 (1466 + 7) |
| Model użyty | `gemini-3.5-flash-lite` |

**Dla porównania:** typowy błąd ręcznego wprowadzania danych w literaturze to
1-5% (Panko i in.), a w bardziej złożonych/niejednorodnych dokumentach nawet
18-40%. Wynik 99,999%/100,000% wyraźnie przewyższa realistyczny scenariusz
ręcznego czyszczenia — i, co ważniejsze, przewyższa też sam LLM działający
bez wsparcia solvera (98,78%/99,52%).

### Diagnoza — jak zmieniała się natura błędów po drodze

Zanim solver powstał, przeanalizowałem błędy samego LLM:

- **91,2%** błędnych dat to była dokładnie zamiana dzień↔miesiąc (oba ≤12) —
  nawet wskazówka kontekstowa (issue_date) czasem nie wystarczała modelowi do
  jednoznacznego rozstrzygnięcia, mimo że matematycznie reguła **jednoznacznie**
  wskazywała poprawną interpretację (zweryfikowane: część "błędów zamiany"
  łamała regułę 1-30 dni w sposób oczywisty, np. odstęp 98 dni zamiast 9 —
  model miał wszystko czego trzeba, i mimo to się mylił).
- Zidentyfikowałem systematyczną słabość modelu przy formacie `MM-DD-YYYY`
  (myślnik) — miesiąc rozpoznawany poprawnie, dzień bywał błędnie kopiowany
  z miesiąca.

To właśnie ta obserwacja — że model *miał* wystarczającą informację i mimo to
zawodził — skłoniła mnie do zbudowania solvera zamiast dalszego poprawiania
promptu.

## Pipeline — jak to działa

Cały pipeline to **jeden plik** (`clean_credit_data_llm.py`), wykonujący 5 kroków
sekwencyjnie po jednym uruchomieniu:

```
dirty_data.csv
      │
      ▼
KROK 1/5: warstwa deterministyczna (kod)  ──────► layer1_cleaned.csv
      │
KROK 2/5: wykrywanie duplikatów (recordlinkage)
      │            │
      │            └──► auto_duplicate.csv / auto_not_duplicate.csv
      │                          │
      └──► duplicate_candidates_gray.csv (strefa szara, ~0,7% par)
                  │
                  ▼
KROK 3/5: warstwa LLM (Gemini, async)  ──► layer2_dates_notes_full.csv
      │                                     + duplicate_verdicts_llm.csv
      │        (pomijalny przełącznikiem SKIP_LLM_RERUN, jeśli wyniki
      │         z poprzedniej sesji już istnieją)
      ▼
KROK 4/5: solver dat (deterministyczny)  ──► nadpisuje daty w layer2
      │        99,95% rekordów rozstrzygniętych bez LLM
      ▼
KROK 5/5: scalenie finalne  ──────► final_cleaned_data.csv  (101 500 wierszy)
```

Warstwa LLM działa **asynchronicznie, paczkami po 100 rekordów**, z
ograniczoną współbieżnością (semafor), regulatorem tempa (rate limiter),
automatycznym ponawianiem przy błędach (w tym walidacją kompletności
odpowiedzi — jeśli model zwróci inną liczbę rekordów niż dostał, paczka jest
automatycznie ponawiana), oraz mechanizmem wznawiania pozwalającym
kontynuować przetwarzanie między sesjami bez powtarzania już zrobionej pracy.

## Struktura repozytorium

```
├── clean_data.csv                    # ground truth (nie używane przez LLM)
├── dirty_data.csv                    # dane wejściowe z kontrolowanym brudem
├── final_cleaned_data.csv            # WYNIK KOŃCOWY
├── dirt_log_ground_truth.csv         # log każdego zabrudzenia (do audytu)
├── duplicate_ground_truth.csv        # mapowanie duplikatów (ground truth)
├── generate_clean_data.py            # generator ground truth
├── dirty_data.py                     # generator kontrolowanego brudu
├── clean_credit_data_llm.py          # GŁÓWNY PIPELINE (5 kroków, 1 plik)
└── data_cleaning_process.ipynb       # proces eksploracji i decyzji
```

## Co dalej / możliwe rozszerzenia

- Osobna walidacja czy pary duplikatów wykryte przez kod nie mają nakładających
  się `id` (drobna rozbieżność 1473 vs 1483 wykryta podczas scalania)
- Rozszerzenie brudzenia o pole `email` (zauważone jako pominięte)
- Dla ~48 naprawdę niejednoznacznych dat: dodatkowy sygnał rozstrzygający
  (np. statystyka większościowa formatu dominującego w danym źródle)

---

*Autor: [LukaszB-DA](https://github.com/LukaszB-DA) · Projekt portfolio — ETL z LLM, czyszczenie danych bankowych na dużą skalę (Python + Gemini API).*
