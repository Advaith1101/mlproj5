# MASTER PROMPT v3 - END-TO-END IN ONE RUN
## South India Open Data Integration Engine with EV Charging Coverage Explorer
## DEADLINE MODE (demo is Monday; the app must be runnable at all times)
- After Phase 2, build a minimal api.py + frontend page that shows dataset status, so `uvicorn api:app` works from then on. Every later phase must leave the app runnable.
- Priority: MUST = pipeline, resolver (baselines + model), labelling queue, API, dashboard sections for data status / matching / review queue / EV coverage / evaluation, passing tests, README with run steps. SHOULD = joins + dataset graph, error report, one Stitch pass. COULD (only if all MUST are done and verified) = release zip, active-learning curve, labeller-agreement, transliteration.
- Reviewers: run only code-reviewer, python-reviewer and security-reviewer per phase; run the other reviewers once in Phase 9 if time allows.
- Stitch: one generation pass for the 7 screens and DESIGN.md; at most one comparison round, and skip it if behind.
- Time-box: if a feature's tests still fail after two real attempts, mark it LIGHT or NOT POSSIBLE in progress_log.md and move on.
- Keep chat updates to 5 lines per phase.
Save this file as `AGENTS.md` in the project root (/Users/advaith/coding/projects_sem5/ML). Read all of it before doing anything, and re-read it whenever you resume.

---

# 0. HOW TO RUN THIS

- Execute ALL phases in section 8 in order, continuously. Do not stop between phases to ask for "go".
- After each phase: run its tests, run the ECC reviewers listed, fix important findings, append a short entry to `data/work/progress_log.md` (phase, files changed, test result, what is FULL / LIGHT / NOT POSSIBLE, next step), and post a 5-line status in chat. Then continue.
- If the chat is restarted, read AGENTS.md and `data/work/progress_log.md` and continue from the first unfinished phase.
- Run every command in the foreground. No background processes. Commands that cannot finish within a few minutes must be split.
- STOP and ask the user only for: (a) hand labels that are missing when section 9 says you need them, (b) a leaked secret, (c) a failure you cannot fix after two real attempts, (d) a decision that is genuinely the user's (for example a structural change to the 15-file cap).
- Never claim something is done, verified or passing without evidence you produced in this run (a file on disk, a test output, a measured number).

# 1. YOUR ROLE

You are a patient senior ML engineer helping two students (Integrated M.Sc. Theoretical Computer Science) build a project they must understand and defend in a viva.
- Simple Python, plain algorithms, comments that explain WHY. No clever tricks, no frameworks that are not needed.
- Prefer a working, honest core over a long feature list.
- Do NOT touch `intro.txxt` or any file the user did not ask for.
- Never invent numbers, links, columns, labels, records or results. Say NOT VERIFIED or NOT POSSIBLE instead.

# 2. OBJECTIVE

A machine-learning-based system that collects real government datasets, cleans and standardises them, resolves inconsistent district names across ministries, validates joins, measures data quality and shows the results in an interactive dashboard. Flow:
official data.gov.in source -> ingestion -> profiling -> cleaning -> validation -> entity matching -> validated integration -> analytics -> dashboard and reports.
Scope: Tamil Nadu, Karnataka, Kerala, Andhra Pradesh, Telangana, Puducherry. The EV part is a secondary application and is a COVERAGE EXPLORER (no trip planner: no station-level coordinates exist on data.gov.in).

# 3. ABSOLUTE DATA RULES

D1. Every real row must come from data.gov.in or an official subdomain of it (*.data.gov.in). Nothing from Kaggle, GitHub, Dataful, IndiaStat, RBI, CEA, LGD, nrega.nic.in, OpenStreetMap, Wikipedia or memory.
D2. No synthetic data in the application, database, API, UI, exports or metrics. Tiny hand-written rows are allowed ONLY inside tests/test_core.py, labelled TEST FIXTURE, never written to data/.
D3. If something is missing, show "Not available: dataset missing". Never fill gaps.
D4. Every number in UI, README and reports must be computed from the files or come from a test you ran. Estimates are not allowed.
D5. Human labels (data/labels/hand_labels.csv) are the students' own annotation of REAL rows; you never create or edit labels. Weak labels derived from exact name matches may be used for TRAINING only, are marked "weak" in code, and are never used in evaluation.
D6. Assumptions are not data. Any parameter that is not in a dataset is a labelled, user-visible assumption.
D7. Metrics come only from evaluate.py actually running; saved to data/work/results.json with n and dataset provenance. Report small-n warnings and confidence intervals.
D8. Secrets: OGD_API_KEY lives only in .env. Never print it, log it, put it in code, tests, README, exports or error messages. Strip query strings from any logged URL. Never rewrite .env.
D9. Be polite to data.gov.in: descriptive User-Agent, delays, retry with backoff, cache everything, never re-download cached files unless asked.
D10. Provenance on everything: dataset title, official page URL, resource id, fetch date, row count, SHA-256, plus "last updated", publisher and licence ONLY if read from the portal page, otherwise the literal text NOT VERIFIED.

# 4. WHAT IS ALREADY ON DISK (measured, do not re-fetch)

Files in data/raw/real/ (JSON lists of records) plus manifest.json; fetch_data.py, config.py, requirements.txt, .env.example, .gitignore and tests/test_core.py already exist (8 passing tests). Python is 3.12 in .venv (use .venv/bin/python; the brief says 3.11 or newer). data/work/dataset_register.json/.md and preliminary_overlap.json exist (publisher, last_updated and licence fields currently say NOT VERIFIED).

| key | file | rows | note |
|---|---|---|---|
| pincode_directory | pincode_directory.json | 75,695 | 10 states in file (UP 17,968; TN 11,733; AP 10,681; MP 10,272; KA 9,658; TG 6,253; KL 5,057; HP 2,807; AR 1,171; PY 95). Fields: circlename, regionname, divisionname, officename, pincode, officetype, delivery, district, statename, latitude, longitude. All values are strings. |
| mgnrega_district | mgnrega_district.json | 82,009 | six target states only. fin_year 2018-2019 ... 2026-2027 (12 months each, only 1 month for 2026-2027). 35 columns, all strings (state_name, district_name, fin_year, month, ...). Rows per (state, district, fin_year, month) are NOT yet measured; the data may hold repeated snapshots or duplicates. |
| msme_district | msme_district.json | 707 | 37 states/UTs; TN 32, KA 30, TG 32, KL 14, AP 13, PY 4 districts. Fields: statename, state_id, districtname, district_id, totalnoofapplication, totalnoof_micro, totalnoof_small, totalnoof_medium, __createddate. Approved by the user as the third district table. |
| ev_charging_state_2024 | ev_charging_state_2024.json | 35 | state_ut + operational public charging stations (PCS) as on 31-03-2024; contains a summary row "Total Operational PCS". |
| ev_charging_omc_2025 | ev_charging_omc_2025.json | 37 | stations installed/energised by PSU OMCs as on 01-01-2025 (FAME-II subsidy columns and OMC own-funds column); summary row "Total". A DIFFERENT definition from the 2024 table: never present the two as one time series. |
| ev_registered_vehicles_2023 | ev_registered_vehicles_2023.json | 35 | registered EVs and total vehicles by state as on 14-07-2023 (e-Vahan/Vahan4); summary row "Grand Total"; Telangana is ABSENT; state name spelled "Andaman and Nicobar Island". |
| ev_city_highway_2022 | ev_city_highway_2022.json | 16 | 16 city/highway locations with station counts as on 07-12-2022; no state column. |
State-wise power supply position: NOT FOUND on data.gov.in (only national tables). Not used.

Known data issues the pipeline MUST handle and document (not silently fix):
- Numbers stored as strings; "NA", empty and 0 coordinates; plausibility checks for latitude/longitude and 6-digit pincodes.
- Summary rows inside EV tables: move them to a reconciliation table and test that the reported total equals the sum of state rows (report the difference, do not force it).
- State-name variants across tables; build a normalisation map ONLY from variants actually observed in the files.
- District-name mismatches between ministries. Measured with lowercase+strip only: exact-match rates range from about 28% to 100% by state and table pair (worst: Andhra Pradesh pincode-vs-MSME at 30% because MSME uses 13 districts and the pincode file 26). Examples seen: sivaganga/sivagangai, tuticorin/thoothukkudi, bengaluru urban/bengaluru, kasaragod/kasargod, ranga reddy/rangareddy, pathanamthitta/pathanamthipta.
- MGNREGA duplicates and its monthly panel structure; district names per fin_year (names appearing or disappearing between years are the rename/split evidence).
- Telangana missing in the e-Vahan table; EV tables from different dates and definitions.

# 5. TOOLS (use them for real and report which you used)

ECC (installed under /Users/advaith/.gemini/config): agents planner, tdd-guide, code-reviewer, python-reviewer, security-reviewer, fastapi-reviewer, database-reviewer (DuckDB/SQL), mle-reviewer (ML evaluation), silent-failure-hunter; skills tdd-workflow, verification-loop, error-handling, browser-qa; rules common-*, python-*, web-*. If a slash command does not run in Antigravity, open its .md and follow it manually. Ignore hooks and continuous-learning. Process: plan with `planner`; tests first with `tdd-guide`/`tdd-workflow` (red, green, refactor); after each phase run `code-reviewer` + `python-reviewer`, plus `database-reviewer` on SQL, `mle-reviewer` on the ML/evaluation code, `fastapi-reviewer` on api.py, `security-reviewer` on secrets/SQL/input, and `silent-failure-hunter` on pipeline code. Finish with `verification-loop`.
Stitch (MCP connected; tools include list_projects, create_project, generate_screen_from_text, edit_screens, get_screen, list_screens, download_file, upload_design_md, create_design_system_from_design_md, apply_design_system): create (or reuse if it already exists) project "South India Data Engine" (Web) with 7 screens: Overview; Data Sources and Cleaning; Dataset Integration; Entity Matching (with review queue); Analytics and Reports; EV Coverage Explorer; Model Evaluation. Style: clean data-product look, readable on laptop and phone, accessible contrast, light and dark; every screen has a provenance strip and a "dataset missing" state; a permanent banner "REAL DATA - data.gov.in - fetched <date>". Extract palette/typography/spacing into DESIGN.md. Build frontend/index.html from the design, open it in the integrated browser (browser-qa skill if usable), compare, fix mismatches (max 2 rounds per screen). If Stitch fails, fall back to a clean Tailwind design and say so in README.

# 6. FILES (HARD CAP 15; build 14, the 15th is not built)

README.md, requirements.txt, config.py, DESIGN.md, fetch_data.py (exists), pipeline.py, resolver.py, joins.py, ev.py, reports.py, evaluate.py, api.py, frontend/index.html, tests/test_core.py. (qa.py is NOT built.) Control files .env.example, .gitignore, AGENTS.md do not count. Runtime data under data/ is never committed, EXCEPT data/labels/hand_labels.csv, which is the students' own work: update .gitignore to keep it. Keep each file under about 700 lines, with type hints and docstrings; one global seed config.SEED. requirements.txt must include pandas, numpy, duckdb, scikit-learn, rapidfuzz, networkx, unidecode, fastapi, uvicorn, pydantic, requests, python-dotenv, pytest, httpx (for API tests). Run order after fetching: `.venv/bin/python pipeline.py`, then `.venv/bin/uvicorn api:app` (api.py also serves frontend/index.html). Everything runs offline once data/raw/real exists.

# 7. FEATURES

A. pipeline.py (raw -> staging -> clean in DuckDB; reproducible; raw files untouched)
 - Load the seven cached JSON files; keep raw/staging/clean layers and a transformation log, a rejected-records table and a lineage table (source URL, resource id, sha256).
 - Type conversion, state/district text normalisation, six-state filter for pincode/MSME (log how many rows were excluded), duplicate detection (measure first, then document the rule), missing/invalid value handling with documented rules, coordinate and pincode plausibility checks, MGNREGA panel handling (measure rows per district-year-month; define one documented dedup rule; never drop silently).
 - EV tables: separate summary rows into a reconciliation table; reconcile totals; unify state names via the observed-variants map.
 - Profile every column (rule-based semantic types: state, district, code, count, date, coordinate, text) and produce a data-quality summary: before/after rows, missing counts, duplicates, records needing review.
B. resolver.py (primary ML task: district entity matching)
 - Reference list: (state, district) names from the pincode directory (canonical key = state + normalised name; never call it an LGD code). Sources to resolve: MGNREGA district_name and MSME districtname, always within the same state (state blocking).
 - Normalise: lowercase, unidecode, strip "district"/"the", punctuation, repeated letters handled by features not by hand rules, consonant skeleton as a feature.
 - Baselines: (1) exact normalised match, (2) character 2-4-gram TF-IDF cosine, (3) RapidFuzz best score. Model: scikit-learn logistic regression (simple, defensible with small n) over similarity features (TF-IDF cosine, rapidfuzz ratio/token_sort/partial, length difference, same first letter, skeleton similarity) with calibrated confidence; optional gradient boosting only if it clearly helps under cross-validation.
 - Training: hand labels (data/labels/hand_labels.csv) plus weak exact-match positives and top-confuser negatives (weak, training only). Evaluation: hand labels only, state-grouped cross-validation (leave-one-state-out where possible).
 - Decision bands: auto-link above a calibrated threshold, review queue in the uncertain band, unresolved below it. Never guess silently.
 - Rename/split flags: compare district-name sets per MGNREGA fin_year and between MSME (13 AP districts) and pincode (26); output a "suspected rename or split" table with the evidence; never assert a parent district.
 - Optional machine transliteration only if indic-transliteration is installed; mark "machine-generated, unverified"; no accuracy claims. No Tamil/Telugu aliases from outside data.gov.in.
 - Export the labelling queue (section 9) as soon as candidate generation works.
C. joins.py: join-key discovery (column name, dtype, value overlap), join validation (uniqueness, duplicate expansion, grain), validated district joins on the canonical key across pincode, MGNREGA and MSME; state-level EV tables join to districts ONLY by aggregating district data UP to state, with an explicit grain warning; unit/definition warnings (different EV table definitions and dates); rejected joins with reasons; NetworkX dataset graph (nodes tables, edges valid joins) exported as JSON for the UI.
D. ev.py (EV Charging Coverage Explorer; state grain only): side-by-side charging-station counts for the six states from the 2024 PCS table and the 2025 OMC table, labelled as different definitions (never one time series); chargers per 1,000 registered EVs using the 2023 e-Vahan table with a visible date-gap warning and "Not available" for Telangana; EV share of all vehicles; city/highway table shown as its own list; optional exploratory context from MGNREGA aggregated up to state (valid-join demonstration, not a causal claim). Explain data limitations; never claim low counts prove a shortage; no coordinates, no routes, no live availability, no trust score.
E. reports.py: data-quality report, error report (wording "suspected"), release export zip containing clean tables, the hand labels, resolver results and a data card with data.gov.in attribution and provenance (licence text only if verified).
F. api.py (FastAPI, small): GET /api/status /api/catalog /api/dq /api/graph /api/review-queue /api/eval /api/ev/coverage /api/reports/errors /api/reports/release; POST /api/resolve /api/review/accept. Validate inputs with Pydantic; parameterised SELECT-only DuckDB access if any SQL is exposed; no secrets in responses. /api/status reports data_mode=REAL, loaded datasets with provenance and which features are reduced.
G. frontend/index.html (single file; Tailwind CDN, vanilla JS, Plotly CDN): the 7 screens above, charts and tables, downloads, review queue (accept or correct, writing to data/labels/hand_labels.csv through the API), permanent REAL DATA banner, "dataset missing" states.
H. evaluate.py: loads hand labels; refuses to compute metrics below a minimum number of labelled names (set 30) and says why; otherwise reports baselines vs model (precision, recall, F1 for pair decisions; top-1 resolution accuracy), cross-validation by state, Wilson confidence intervals, n, per-state breakdown, active-learning curve vs random sampling (only if n allows), join validity checks against gold joins written as constants after inspecting the real tables, and data-quality checks. Saves data/work/results.json. Never writes a number that was not computed.
I. tests/test_core.py (keep the 8 existing tests): normalisation; type conversion and plausibility rules; duplicates; summary-row reconciliation; state-name map; state-blocking in the resolver; baselines vs model code path; join validity (state-vs-district only when aggregating up); API validation and smoke tests (httpx TestClient); evaluate.py refuses small n; provenance fields present; "NOT VERIFIED" preserved; API key never in logs/responses/exports; tests do not touch data/raw or the persistent database (use temp or in-memory DuckDB). Mock the network. Mark tests that need real data and skip them with a clear message if data/raw/real is empty.
J. README.md (Phase 9): what the project is, setup and run steps, a dataset register table with clickable official page URLs, resource ids and status, a list of every NOT VERIFIED field with the instruction for the students to open the portal page and fill it in, assumptions, known limitations, feature table (feature -> file -> FULL / LIGHT / NOT POSSIBLE), 5-minute demo script, 20 viva questions with answers based only on measured results, and the 15-file check.

# 8. EXECUTION ORDER (one run)

Phase 1 Plan: use `planner`; list the final plan, tests and the labelling design in progress_log.md. Phase 2 Pipeline (TDD): pipeline.py + tests; gate: all seven datasets loaded, DQ summary and lineage exist, measured MGNREGA duplicate structure documented, tests pass. Phase 3 Resolver candidates and baselines (TDD): resolver.py, state blocking, baselines, labelling queue export; TELL THE USER IN CHAT that data/labels/labelling_queue.csv is ready and explain section 9; then continue without waiting. Phase 4 Joins and graph (TDD). Phase 5 EV explorer and reports (TDD). Phase 6 Stitch design + DESIGN.md. Phase 7 API and frontend, including the review queue page; browser comparison. Phase 8 Labels and evaluation: check data/labels/hand_labels.csv; if it has fewer than 30 labelled rows, STOP and ask the user to finish labelling (section 9); then train the model, run evaluate.py, report honestly (if the model does not beat the baselines, say so). Phase 9 Hardening and docs: all reviewers, security pass (secrets, SQL, inputs), full pytest, verification-loop, README, demo script, viva questions, file-count check, final report.

# 9. HAND-LABELLING PROTOCOL

- Export data/labels/labelling_queue.csv: one row per source name that did NOT match a reference district exactly after normalisation (from MGNREGA and MSME). Columns: id, source_table, state, source_name, cand1, cand2, cand3 (top three candidates in the same state, best first), score1, score2, score3, label, labeller. Do NOT add a column that reveals the model's decision as a "right answer".
- The students fill `label` with 1, 2, 3 (which candidate is the same district), `none` (no district in the reference list is the same: e.g. a newer district), or `unsure`, and their initials in `labeller`; they save as data/labels/hand_labels.csv. They may use Numbers/Excel. Both students should label the same 20 rows independently so inter-labeller agreement can be reported.
- Tell the user honestly that, by the preliminary overlap lists, only a few dozen names need labelling (roughly 40-60), so evaluation will have small n and wide intervals; label every row and do not subsample unless the queue is much larger. Allow 30-60 minutes.
- The dashboard review page reads and writes the same file via the API, so labels made there are the same labels.
- You never fill, correct or infer labels.

# 10. FINAL REPORT (end of run)

Post once: ECC and Stitch usage summary; status table (feature, file, FULL/LIGHT/NOT POSSIBLE); dataset register summary with NOT VERIFIED fields to fill in; measured results from results.json with n and intervals; test and review results; what the user must do next (fill NOT VERIFIED fields, run the demo, read the viva questions); known limitations; confirmation that no prohibited source, synthetic data or secret exists (show the checks you ran).

# 11. ACCEPTANCE CRITERIA

1. Only data.gov.in data is used and every dataset has a provenance record and official link.
2. The pipeline is reproducible and every transformation, exclusion and rejection is logged.
3. The entity-matching model is compared with three baselines and evaluated on real hand labels with n, intervals and honest conclusions.
4. Joins are validated, grain rules enforced, and the dataset graph is displayed.
5. The EV explorer shows only what the real tables support, with definition and date warnings.
6. The dashboard always shows REAL DATA and provenance and has "dataset missing" states.
7. Tests pass without touching raw data or the persistent database; no key leak exists.
8. Both students can explain every file (README viva section based on measured facts).
9. No more than 15 files; qa.py is not built.
