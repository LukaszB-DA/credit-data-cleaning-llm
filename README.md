# ETL z LLM — inteligentne czyszczenie danych bankowych na dużą skalę

## O projekcie

Drugi etap mojego portfolio z zakresu czyszczenia danych — po ręcznym, klasycznym
czyszczeniu danych kredytowych (Excel/Power Query + Python), ten projekt pokazuje
**świadome** wykorzystanie LLM w pipeline ETL: nie "wrzucam wszystko do modelu bo
się da", tylko architekturę, która używa taniego, deterministycznego kodu tam,
gdzie to wystarcza, a LLM tylko tam, gdzie faktycznie daje przewagę.

**Cel:** oczyścić 101 500 rekordów bankowych (kredyty, styl Lending Club) w jednym,
zautomatyzowanym przebiegu, łącząc klasyczne techniki (pandas, regex, fuzzy
matching, rekord linkage) z modelem językowym (Gemini) tam, gdzie reguły
zawodzą — niejednoznaczne formaty dat i wolny tekst.

> **Uwaga językowa:** prompty wysyłane do modelu oraz komunikaty w `print()`
> są napisane po polsku — to był świadomy wybór na etapie budowy, nie
> przeoczenie. Komentarze w kodzie są sukcesywnie uzupełniane wersją
> angielską, docelowo repo będzie miało dwujęzyczne README (PL/EN).

## Dane

Ponieważ realne dane bankowe (np. pełny zbiór Lending Club) nie były dostępne
w wystarczająco "surowej" formie (dostępne kopie były już wstępnie zakodowane/
oczyszczone, bez pól tekstowych), zbudowałem:

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

**Eksploracja danych:** zanim napisałem reguły czyszczące, przeglądałem dane
iteracyjnie w Data Wrangler (VS Code/Jupyter) — filtry, `groupby`, sortowanie
ASC/DESC — do szybkiego wyłapywania outlierów, wartości nielogicznych i
wzorców w brudnych kolumnach, zanim to przełożyło się na konkretne reguły w
kodzie (np. `pd.to_numeric(errors="coerce")` do wykrywania niesparsowalnych
wartości, regex przy ekstrakcji `grade`).

## Architektura — dwie warstwy

### Warstwa 1: kod (deterministyczny, zero LLM)

Obsługuje wszystko, co da się rozwiązać regułami/dopasowaniem tekstu:

| Problem | Metoda | Skuteczność |
|---|---|---|
| Braki danych (5 zapisów) | `.replace()` na markery + `pd.to_numeric(errors="coerce")` do wykrywania | 100% |
| Separator dziesiętny | `pd.to_numeric(errors="coerce")` | 100% |
| Wartości fizycznie niemożliwe | reguły zakresowe | 100% |
| `home_ownership`, `purpose` (literówki) | fuzzy matching (rapidfuzz) | 100% |
| Telefon | ekstrakcja cyfr + reformatowanie | 100% |
| Duplikaty klientów | blokowanie + porównanie (recordlinkage) | 99,3% par ze 100% trafnością |

### Warstwa 2: LLM (Gemini 3.5 Flash-Lite) — tylko eskalacja

Trafia tam **tylko** to, czego warstwa 1 nie rozstrzygnie jednoznacznie:

- **Daty** (`application_date`, `issue_date`) — 5 formatów wejściowych,
  niejednoznaczność dzień/miesiąc. Model wykorzystuje kontekst (issue_date
  zawsze 1-30 dni po application_date) do rozstrzygania.
- **`advisor_notes`** — wolny tekst, literówki, normalizacja stylu.
- **Strefa szara duplikatów** — ~0,7% par, gdzie kod nie jest pewny.

**Uzasadnienie tego podziału:** przed użyciem LLM zmierzyłem skuteczność
klasycznego parsera dat (lista znanych formatów) — wyszło **~61% poprawnych
dat**, z czego 3657 przypadków to *ciche* pomyłki dzień/miesiąc (parser nie
rzucał błędu, po prostu dawał złą datę). To pokazało, że akurat to zadanie
faktycznie potrzebuje LLM.

## Wyniki końcowe (101 500 rekordów)

| Metryka | Wynik |
|---|---|
| `application_date` | 98,78% |
| `issue_date` | 99,52% |
| Duplikaty (kod) | 1466 par, 100% trafność |
| Duplikaty (LLM, strefa szara) | 17/17 poprawnych ocen |
| Model użyty | `gemini-3.5-flash-lite`, jeden na całości |

**Dla porównania:** typowy błąd ręcznego wprowadzania danych w literaturze to
1-5% (Panko i in.), a w bardziej złożonych/niejednorodnych dokumentach nawet
18-40%. Wynik 98,78%/99,52% jest porównywalny lub lepszy niż realistyczny
scenariusz ręcznego czyszczenia, przy ułamku czasu.

### Diagnoza pozostałych błędów dat

Nie poprzestałem na samej metryce — sprawdziłem **naturę** błędów:

- **91,2%** błędnych dat to dokładnie zamiana dzień↔miesiąc (oba ≤12) — nawet
  wskazówka kontekstowa (issue_date) czasem nie wystarcza do jednoznacznego
  rozstrzygnięcia.
- Zidentyfikowałem też systematyczną słabość modelu przy formacie
  `MM-DD-YYYY` (myślnik) — miesiąc rozpoznawany poprawnie, ale dzień bywał
  błędnie kopiowany z miesiąca.
- Pozostałe ~9% błędów (w tym kilka `null` mimo jednoznacznych dat) to
  pojedyncze przypadki bez wspólnego wzorca.

## Pipeline — jak to działa

Cały pipeline to **jeden plik** (`clean_credit_data_llm.py`), wykonujący 4 kroki
sekwencyjnie po jednym uruchomieniu:

```
dirty_data.csv
      │
      ▼
KROK 1/4: warstwa deterministyczna (kod)  ──────► layer1_cleaned.csv
      │
KROK 2/4: wykrywanie duplikatów (recordlinkage)
      │            │
      │            └──► auto_duplicate.csv / auto_not_duplicate.csv
      │                          │
      └──► duplicate_candidates_gray.csv (strefa szara, ~0,7% par)
                  │
                  ▼
KROK 3/4: warstwa LLM (Gemini, async)  ──► layer2_dates_notes_full.csv
      │                                     + duplicate_verdicts_llm.csv
      ▼
KROK 4/4: scalenie finalne  ──────► final_cleaned_data.csv  (101 500 wierszy)
```

Warstwa LLM działa **asynchronicznie, paczkami po 100 rekordów**, z
ograniczoną współbieżnością (semafor), regulatorem tempa (rate limiter),
automatycznym ponawianiem przy błędach (w tym walidacją kompletności
odpowiedzi — jeśli model zwróci mniej rekordów niż dostał, paczka jest
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
├── clean_credit_data_llm.py          # GŁÓWNY PIPELINE (4 kroki, 1 plik)
└── data_cleaning_process.ipynb       # proces eksploracji i decyzji
```

## Co dalej / możliwe rozszerzenia

- Osobna walidacja czy pary duplikatów wykryte przez kod nie mają nakładających
  się `id` (drobna rozbieżność 1473 vs 1483 wykryta podczas scalania)
- Test skuteczności modelu na formacie `MM-DD-YYYY` z dodatkową heurystyką
- Rozszerzenie brudzenia o pole `email` (zauważone jako pominięte)

---

*Autor: [LukaszB-DA](https://github.com/LukaszB-DA) · Projekt portfolio — ETL z LLM, czyszczenie danych bankowych na dużą skalę (Python + Gemini API).*
