# South India Open Data Integration Engine with EV Charging Coverage Explorer

**Academic Level:** Third-Year Integrated M.Sc. Theoretical Computer Science  
**Scope:** Tamil Nadu, Karnataka, Kerala, Andhra Pradesh, Telangana, Puducherry  
**Mandatory Data Source:** Strictly [data.gov.in](https://www.data.gov.in/) (Official Open Government Data)  
**Primary Focus:** Data cleaning, entity matching under severe ministry variation, validated integration, and EV coverage analytics.

---

## 1. Project Overview & Architecture

Government open data in India is published independently by distinct ministries without shared primary keys, harmonized district nomenclatures, or common temporal grain. This project is a defensible, machine-learning-driven data standardisation and integration engine that:
1. Ingests raw data.gov.in datasets into a DuckDB persistent mart across layered schemas (`raw_*`, `staging_*`, `clean_*`, `meta_*`).
2. Enforces data plausibility rules for postal codes (6 digits, valid Indian prefix), geographic coordinates (India bounding box ~6–38°N, 68–98°E), and deduplicates monthly panel records.
3. Performs district entity resolution using strict state-level blocking, consonant skeleton extraction, character n-gram TF-IDF cosine similarity, RapidFuzz token sorting, and calibrated Logistic Regression.
4. Enforces strict grain hierarchy during multi-table joins: state-level EV tables can only join to district indicators by aggregating district data **UP** to state grain (prohibiting spurious duplicate expansion).
5. Explores state-level EV charging infrastructure coverage side-by-side with definition divergence and date gap warnings.

---

## 2. Setup and Execution Guide

### Prerequisites
- Python 3.11 or newer (Python 3.12 verified)
- macOS / Linux / Windows

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/Advaith1101/mlproj5.git
cd mlproj5

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install exact pinned dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment Secret
```bash
# Copy example environment configuration
cp .env.example .env

# Edit .env and supply your data.gov.in API key (kept strictly private; never committed)
# OGD_API_KEY=your_key_here
```

### Step 3: Run Ingestion & Integration Pipeline
```bash
# Executes pipeline.py: loads cached JSONs, filters 6 states, validates types, builds DuckDB mart
python pipeline.py --db data/work/engine.duckdb

# Executes joins.py: builds district & state integration marts, exports NetworkX graph
python joins.py
```

### Step 4: Run Tests
```bash
# Run complete test suite (39 passing unit tests)
pytest tests/test_core.py
```

### Step 5: Launch Interactive Dashboard
```bash
# Launch FastAPI server serving the 7-screen single-page dashboard
uvicorn api:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

## 3. Dataset Register & Provenance

Every record originates strictly from official portals under data.gov.in.

| Key | Title | Ministry / Dept | Official Resource ID | Raw Rows | Clean Rows | Official Portal Link |
|---|---|---|---|---|---|---|
| `pincode_directory` | All India Pincode Directory | Dept of Posts | `6176ee09-3d56-4a3b-8115-21841576b2f6` | 75,695 | 43,477 | [Portal Link](https://www.data.gov.in/resource/all-india-pincode-directory-till-last-month) |
| `mgnrega_district` | MGNREGA Monthly Employment | Min. of Rural Development | `eeab5eb3-9764-4286-9dc5-dcae8ca47195` | 82,009 | 12,193 | [Portal Link](https://www.data.gov.in/resource/district-wise-monthly-employment-generated-under-mgnrega) |
| `msme_district` | MSME Udyam Enterprises | Min. of Micro, Small & Medium | `c06497f2-1b1f-49ff-813c-4ef1e0d45362` | 707 | 125 | [Portal Link](https://www.data.gov.in/resource/state-and-district-wise-number-enterprises-applied-under-udyam-registration) |
| `ev_charging_state_2024` | Operational Public Charging Stations | Min. of Heavy Industries | `1d977457-36e4-4d8b-bb8c-7f5fbca41c1e` | 35 | 34 | [Portal Link](https://www.data.gov.in/resource/stateut-wise-details-operational-public-charging-stations-pcs-31032024) |
| `ev_charging_omc_2025` | OMC Retail Outlet EVCS | Min. of Petroleum & Natural Gas | `3bc8565a-be18-472e-84dd-2436ba45ec77` | 37 | 36 | [Portal Link](https://www.data.gov.in/resource/state-wise-details-ev-charging-stations-installedenergized-oil-marketing-companies-01012025) |
| `ev_registered_vehicles_2023`| Registered Electric & Total Vehicles | MoRTH (e-Vahan) | `62402127-b649-411a-8280-993d052cb443` | 35 | 34 | [Portal Link](https://www.data.gov.in/resource/state-wise-data-electric-vehicles-and-total-vehicles-registered-14072023) |
| `ev_city_highway_2022` | Highway & Expressway Corridors | MoRTH | `93ca4b4c-9f79-43c2-a42e-1311b93ea300` | 16 | 16 | [Portal Link](https://www.data.gov.in/resource/details-operational-charging-stations-various-expressway-and-highways-across-country-07122022) |

### Unverified Portal Fields (Action for Students)
Per Rule D10, fields not directly readable on raw API payloads are set to `NOT VERIFIED`. Students should open the official portal page URLs above and verify:
1. `last_updated` date string on the portal resource overview card.
2. `publisher` exact department title.
3. `licence` (confirming "Government Open Data License - India (GODL)").

---

## 4. Key Engineering & Machine Learning Features

### 4.1 Strict State-Level Blocking
Cross-ministry district entity resolution is strictly partitioned by state. When matching a district in Tamil Nadu, candidates are restricted to Tamil Nadu reference districts from the Department of Posts. This prevents catastrophic cross-state errors (e.g. matching `dharwad` in Karnataka to a similarly named town in Madhya Pradesh).

### 4.2 Orthographic & Phonetic Feature Extraction
- **Consonant Skeleton:** Eliminates vowels (`a, e, i, o, u`), non-alphanumerics, and collapses consecutive identical consonants. E.g. `bengaluru` → `bnglr`, `kasaragod` → `ksrgd`, `chennai` → `chn`.
- **Composite Feature Vector:**
  1. `rf_ratio`: Levenshtein ratio.
  2. `rf_token_sort`: Word-order invariant token sorting.
  3. `rf_partial`: Substring alignment score.
  4. `skeleton_sim`: Consonant skeleton similarity.
  5. `first_char_match`: Binary match on initial character.
  6. `len_diff`: Absolute character length disparity.

### 4.3 3 Baselines vs ML Model
- **Baseline 1:** Exact Normalized String Match.
- **Baseline 2:** Character 2-4-gram TF-IDF Cosine Similarity.
- **Baseline 3:** RapidFuzz Token Sort Ratio.
- **Model:** Scikit-Learn Logistic Regression with calibrated probability bands:
  - *Auto-Link:* Confidence $\ge 0.85$
  - *Review Queue:* $0.50 \le \text{Confidence} < 0.85$
  - *Unresolved:* $\text{Confidence} < 0.50$

### 4.4 Hand-Labelling Protocol & Small-n Guardrail
The candidate generator identified exactly 36 non-exact district variations from MGNREGA and MSME. Students annotate these in `data/labels/hand_labels.csv` using options `1`, `2`, `3`, `none`, or `unsure`.  
`evaluate.py` strictly enforces a guardrail: it **refuses** to compute statistical metrics if valid labelled rows are below 30, preventing misleading claims on unrepresentative subsets. All accuracy metrics report Wilson 95% confidence intervals.

---

## 5. EV Charging Coverage Explorer (Option A)

Government data provides state-level and highway-corridor charging station counts. Station-level GPS coordinates do not exist on data.gov.in. Consequently, Option A (Coverage Explorer) was implemented instead of a fictional trip planner.

### Explicit Limitations & Integrity Warnings
1. **Definition Divergence Warning:** PCS 2024 measures Public Charging Stations under the Ministry of Heavy Industries. OMC 2025 measures stations installed by PSU Oil Marketing Companies at retail petrol pumps. They represent different definitions and are **never** presented as a single time series.
2. **Date Gap Warning:** e-Vahan vehicle counts are as of 14-07-2023, while chargers are as of 31-03-2024 and 01-01-2025. Coverage density ratios are indicative approximations across differing reporting dates.
3. **Missing State Handling:** Telangana was absent from the e-Vahan 2023 portal file. Per Rule D3, it is displayed as `Not available: dataset missing` (Hypothesis: Telangana operated a separate state transport database during this window).
4. **No Station Coordinates:** No station-level GPS coordinates exist on data.gov.in. Route planning and availability tracking cannot be honestly claimed.

---

## 6. Feature Implementation Status

| Feature | Primary File | Status | Notes |
|---|---|---|---|
| Raw-to-Clean DuckDB Pipeline | `pipeline.py` | **FULL** | Raw JSON cache untouched; 4-layer DuckDB architecture |
| State & District Normalisation | `pipeline.py`, `resolver.py` | **FULL** | Unidecode, strip noise, observed variants map |
| Plausibility Validation | `pipeline.py` | **FULL** | 6-digit PIN, India lat/lon bounding box (~6–38°N, 68–98°E) |
| Summary Row Reconciliation | `pipeline.py`, `api.py` | **FULL** | PCS 2024, OMC 2025, Vahan 2023 verified arithmetic reconciliation |
| MGNREGA Deduplication | `pipeline.py` | **FULL** | 69,816 snapshot duplicates filtered by max persondays/expenditure rule |
| 3 Resolver Baselines | `resolver.py` | **FULL** | Exact, Char n-gram TF-IDF, RapidFuzz Token Sort |
| Logistic Regression Resolver | `resolver.py` | **FULL** | Scikit-learn with state blocking & probability bands |
| Hand-Labelling Queue & UI | `resolver.py`, `frontend/index.html` | **FULL** | 36 candidate rows; live web review queue updating CSV |
| Join Validation & Grain Enforcement | `joins.py` | **FULL** | Enforces upward aggregation for state EV joins; rejects invalid keys |
| NetworkX Dataset Graph | `joins.py` | **FULL** | 9 nodes, 7 validated edges exported to `dataset_graph.json` |
| EV Coverage Comparative Explorer | `ev.py`, `frontend/index.html` | **FULL** | Side-by-side PCS 2024 vs OMC 2025, Plotly charts, corridor view |
| Suspected Error Register | `reports.py` | **FULL** | 5 documented official data anomalies with neutral phrasing |
| Release Package ZIP Export | `reports.py` | **FULL** | Packages clean CSVs, data card, and error report |
| 7-Screen Web Dashboard | `frontend/index.html` | **FULL** | Permanent REAL DATA banner, Tailwind CSS, Plotly CDN |
| Statistical Evaluator | `evaluate.py` | **FULL** | 30-label guardrail, Wilson score 95% confidence intervals |
| Unit Test Suite | `tests/test_core.py` | **FULL** | 39 automated unit tests passing in ~7.5s |

---

## 7. 14-File Hard Cap Compliance

The project adheres strictly to the 14-file architectural cap (Section 6). No unnecessary frameworks or microservices were introduced.

1. `README.md` (Project documentation, run steps, viva defense script)
2. `requirements.txt` (Pinned library dependencies)
3. `config.py` (Paths, target states, dataset manifest, secret hygiene)
4. `DESIGN.md` (Design system tokens, typography, 7-screen architecture)
5. `fetch_data.py` (data.gov.in fetcher, pagination, retry backoff, SHA-256)
6. `pipeline.py` (DuckDB ingestion, type casting, filtering, reconciliation)
7. `resolver.py` (State blocking, baselines, ML features, queue generator)
8. `joins.py` (Grain validation, integration mart, NetworkX graph)
9. `ev.py` (State-grain comparative analytics, limitation warnings)
10. `reports.py` (Data quality audits, suspected errors, release zip)
11. `evaluate.py` (Statistical evaluation, small-n refusal, Wilson intervals)
12. `api.py` (FastAPI backend serving endpoints and dashboard)
13. `frontend/index.html` (7-screen single-page dashboard with review queue)
14. `tests/test_core.py` (39 passing unit tests covering all modules)
*(Control files `.env.example`, `.gitignore`, `AGENTS.md` and pre-existing `intro.txxt` do not count towards the 14 application files. File 15 `qa.py` was explicitly not built).*

---

## 8. 5-Minute Viva Demonstration Script

1. **Minute 1: Ingestion & Integrity**
   - Open terminal: show `.venv/bin/pytest tests/test_core.py` (39 tests pass).
   - Explain that data comes 100% from data.gov.in. Show `clean_pincode` (43,477 rows) and how 32,218 non-South rows were filtered out.
2. **Minute 2: Data Cleaning & Reconciliation**
   - In the dashboard at `http://127.0.0.1:8000/`, open **Tab 2 (Sources & Cleaning)**.
   - Show the EV Summary Reconciliation audit: demonstrate that the reported summary row equals the sum of state rows across all 3 EV tables.
   - Show how MGNREGA monthly revisions were deduplicated (reducing 82,009 raw panel rows to 12,193 clean monthly snapshots).
3. **Minute 3: Entity Resolution & State Blocking**
   - Open **Tab 4 (Entity Matching & Review Queue)**.
   - Type `tuticorin` in Tamil Nadu into the interactive tester: demonstrate that Baseline 1 (Exact) fails, while RapidFuzz and the ML feature vector identify `thoothukkudi tuticorin` with high confidence.
   - Show the student review queue table with the 36 candidate rows.
4. **Minute 4: Dataset Integration & Grain Enforcement**
   - Open **Tab 3 (Integration & Graph)**.
   - Explain why state EV tables can never be joined directly 1-to-many into district rows without duplicate inflation. Show the NetworkX graph demonstrating upward aggregation from district to state grain.
5. **Minute 5: EV Coverage Explorer & Honest Reporting**
   - Open **Tab 5 (EV Coverage Explorer)**.
   - Point out the two prominent warning cards: explain why PCS 2024 and OMC 2025 cannot be combined into a time series, why registered vehicle counts carry a date gap, and why Telangana displays *Not available: dataset missing*.
   - Show **Tab 6 (Reports & Release)**: demonstrate the suspected error report and download the release package zip.

---

## 9. 20 Viva Defense Questions & Measured Answers

**Q1: Why did you not scrape station-level charging locations from OpenStreetMap, Google Maps, or PlugShare?**  
*Answer:* Rule D1 strictly prohibits non-official sources. Data on data.gov.in contains no GPS station coordinates for public charging points. Using third-party scraping would violate official provenance and academic defensibility.

**Q2: What is the exact difference between the 2024 and 2025 EV charging datasets?**  
*Answer:* The 2024 dataset (`ev_charging_state_2024`) measures Operational Public Charging Stations (PCS) under the Ministry of Heavy Industries. The 2025 dataset (`ev_charging_omc_2025`) measures charging points installed by public sector Oil Marketing Companies (IOCL, BPCL, HPCL) at retail petrol pumps. They represent distinct administrative definitions and cannot be merged into a single time series.

**Q3: Why does Telangana show "Not available" in the EV vehicle metrics?**  
*Answer:* Telangana is completely absent from the official data.gov.in e-Vahan dataset as of 14-07-2023 (`ev_registered_vehicles_2023.json`), which lists only 34 states/UTs plus a summary row. Our hypothesis is that Telangana operated an independent state transport database during that period. In accordance with Rule D3, we report the truth of the dataset rather than interpolating synthetic figures.

**Q4: How did you handle summary rows embedded in government tables?**  
*Answer:* Tables like `ev_charging_state_2024` contained a summary row labelled "Total Operational PCS" (12,146). The pipeline extracts these rows into `meta_ev_reconciliation` and arithmetically compares the reported total with the sum of individual state rows, verifying that the difference is 0 without forcing it.

**Q5: Why did the MGNREGA dataset shrink from 82,009 to 12,193 rows?**  
*Answer:* MGNREGA records represent monthly cumulative employment revisions. Multiple entries existed for identical `(state_name, district_name, fin_year, month)` tuples representing progressive data submissions. We applied a documented deduplication rule taking the latest cumulative record with maximum persondays/expenditure, suppressing 69,816 revision duplicates.

**Q6: Why did you use the Pincode Directory as the canonical district reference list?**  
*Answer:* The Department of Posts Pincode Directory provides a granular, ground-level administrative mapping of post offices and PIN codes to districts across all states. We canonicalized districts on `(norm_state, norm_district)` pairs. We never falsely call these "LGD codes" because official Local Government Directory codes are absent from the file.

**Q7: Why does Andhra Pradesh have a severe mismatch between MSME and Pincode tables?**  
*Answer:* Andhra Pradesh underwent a major administrative reorganisation in April 2022, expanding from 13 to 26 districts. The MSME Udyam table reflects the older 13-district structure, whereas the Pincode Directory reflects 26 districts. The resolver flags this as a suspected reorganisation divergence rather than forcing an artificial 1:1 parent mapping.

**Q8: What is strict state-level blocking in the entity matching resolver?**  
*Answer:* Before comparing district names, the search space is blocked strictly by state. When resolving a district from Tamil Nadu, candidates are drawn exclusively from Tamil Nadu reference districts. This reduces candidate comparisons by >80% and eliminates false cross-state matches.

**Q9: What is a consonant skeleton and why is it useful for Indian place names?**  
*Answer:* Indian district names transliterated into English suffer from vowel variation (e.g. `Kasaragod` vs `Kasargod`, `Davanagere` vs `Davangere`). Consonant skeleton extraction strips vowels and collapses repeated consonants (`ksrgd`, `dvngr`), providing a robust phonetic feature invariant to English vowel spelling discrepancies.

**Q10: Why did you use Logistic Regression instead of deep neural networks or LLMs?**  
*Answer:* With a small number of unresolved names (36 candidates), deep learning models overfit and lack mathematical defensibility. A calibrated Logistic Regression model operating on 6 orthographic and phonetic similarity features is transparent, fast, interpretable, and defensible in a viva.

**Q11: What are the three resolver decision bands?**  
*Answer:* Auto-link for confidence $\ge 0.85$, Review Queue for uncertain matches ($0.50 \le \text{confidence} < 0.85$), and Unresolved for confidence $< 0.50$. The system never guesses silently in ambiguous regions.

**Q12: Why does evaluate.py refuse to compute metrics when hand labels are fewer than 30?**  
*Answer:* Computing precision, recall, and accuracy on very small samples ($n < 30$) produces wide, statistically meaningless confidence intervals. The 30-label minimum is a mandatory guardrail ensuring defensible sample size.

**Q13: Why are Wilson score confidence intervals reported instead of standard normal approximations?**  
*Answer:* Standard Wald normal approximation intervals ($\hat{p} \pm 1.96\sqrt{\hat{p}(1-\hat{p})/n}$) fail severely when sample size $n$ is modest or when accuracy $\hat{p}$ is close to 1.0, often producing intervals exceeding 100%. The Wilson score interval properly handles asymmetric error bounds.

**Q14: Why can state EV tables not be directly joined to district tables?**  
*Answer:* Grain mismatch. District tables exist at the district grain, while EV tables exist at the state grain. Performing a direct 1:N join copies the state charger total across every district in that state, falsely creating duplicate expansion and misleading analytics. Our pipeline mandates aggregating district metrics **UP** to state level before joining.

**Q15: Why is the 2022 Highway Corridors table kept isolated from the dataset graph?**  
*Answer:* The corridor table contains 16 highway/expressway names (e.g., Delhi-Jaipur, Mumbai-Pune) but lacks state or district administrative keys. Attempting to join it to state tables would require geographic guesswork, violating Rule D1 and D6.

**Q16: How are coordinates validated in the cleaning pipeline?**  
*Answer:* Coordinate fields stored as strings are checked for 'NA', empty strings, and 0.0 values. Plausibility checks enforce India's geographic bounding box: latitude must lie between 6.0°N and 38.0°N, and longitude between 68.0°E and 98.0°E.

**Q17: What clerical error was discovered in the official MSME table?**  
*Answer:* Pathanamthitta district in Kerala was entered as `pathanamthipta` in the official MSME portal file (`c06497f2-1b1f-49ff-813c-4ef1e0d45362`). Our consonant skeleton and RapidFuzz matchers successfully linked it to `pathanamthitta` (score 0.9324).

**Q18: Why did you use DuckDB as the embedded database engine?**  
*Answer:* DuckDB is columnar, embedded (zero external server dependencies), supports standard ACID SQL and analytical window functions, integrates natively with Pandas DataFrames and Arrow, and executes queries locally in milliseconds.

**Q19: How did you ensure the OGD_API_KEY was never leaked?**  
*Answer:* The API key lives exclusively in `.env`, which is untracked in `.gitignore`. All logged URLs are passed through `sanitize_url()` to strip query parameters, no API responses include secret keys, and unit tests in `test_core.py` assert that secrets do not appear in manifest exports or logs.

**Q20: Why are weak labels prohibited from evaluation?**  
*Answer:* Weak labels derived from exact string matching can provide positive training pairs for the classifier, but using them in evaluation would artificially inflate performance metrics (circular evaluation). Evaluation is conducted strictly on student hand-annotated labels.
