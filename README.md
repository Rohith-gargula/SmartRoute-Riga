

\# SmartRoute-Riga

\## An Explainable Machine Learning Framework for Short-Term Public Transport Demand Forecasting and Operational Planning in Riga



\### Project Overview



SmartRoute-Riga is a machine learning research project that investigates whether historical QR ticket validation activity can be used to forecast the number of QR validations on Riga public transport routes in the next hour.



The project uses publicly available data from Rīgas Satiksme, published through the Latvian Open Data Portal. It applies data quality assessment, feature engineering, time-aware model evaluation, Random Forest regression and error analysis.



\*\*Important measurement limitation:\*\* The dataset records QR ticket validations, not every passenger boarding or travelling on public transport. Therefore, QR validation counts are used as a proxy for observed validation activity and must not be interpreted as total passenger demand.



\### Research Question



Can machine learning predict the number of ticket validations on Riga public transport routes in the next hour using historical validation activity, time patterns, route information and transport characteristics?



\### Dataset



\- Source: Latvian Open Data Portal

\- Publisher: Rīgas Satiksme

\- Dataset: QR code validation data on Riga public transport vehicles

\- Data period: October 2025 – October 2026

\- Transport modes: Autobuss, Trolejbuss and Tramvajs

\- Original records: 533,195

\- Licence: CC0 1.0



Official dataset:

https://data.gov.lv/dati/dataset/kvadrat-kodu-validaciju-dati-rigas-satiksmes-sabiedriskajos-transportlidzeklos



\### Data Preparation



The project includes:



1\. Combining 13 monthly CSV files.

2\. Inspecting data types, missing values and duplicate records.

3\. Auditing transport modes, routes and hourly intervals.

4\. Assessing calendar-date and hourly coverage.

5\. Creating historical features for next-hour forecasting.

6\. Excluding one route with insufficient consecutive observations for reliable next-hour forecasting.



The final modelling dataset contains 499,348 records and 89 transport-route combinations.



\### Methodology



The prediction task is formulated as a supervised regression problem.



The target is the number of QR validations in the next hour for a given transport mode and route.



Features include:



\- Current-hour QR validations.

\- Previous-hour QR validations.

\- Same-hour validations from the previous day.

\- Same-hour validations from the previous week.

\- Forecast hour, day, month and weekend indicators.

\- Cyclical time features.

\- Transport and route information.



\### Models



The following approaches were evaluated:



\- Persistence baseline: the current observed validation count is used as the next-hour prediction.

\- Random Forest Regressor.

\- Extra Trees Regressor.



A chronological split was used to reduce temporal data leakage:



\- Training: before 1 July 2026.

\- Validation: July–August 2026.

\- Testing: September–October 2026.



Random Forest was selected based on validation MAE and evaluated on the held-out test period.



\### Final Test Results



| Metric | Random Forest | Persistence Baseline |

|---|---:|---:|

| MAE | 8.44 | 21.97 |

| RMSE | 16.11 | 50.21 |

| R² | 0.9740 | 0.7478 |

| WAPE | 14.72% | 38.33% |



The Random Forest achieved approximately 61.6% lower MAE than the persistence baseline on the test period.



These are retrospective test results, not a guarantee of future operational performance.



\### Explainability and Error Analysis



Feature importance analysis identified current-hour QR validations and historical same-hour patterns as important predictive signals.



Additional analysis examined:



\- Prediction errors by transport mode.

\- Prediction errors by route.

\- Hourly model performance.

\- Peak and off-peak periods.

\- Prediction bias and largest errors.

\- Tram route 1 at 17:00, including weekday-level comparisons.



The route-level investigation identified consistent underprediction during several weekday 17:00 observations. This highlights the importance of examining errors beyond aggregate model metrics.



\### Project Structure



```text

SmartRoute-Riga/

├── data/

│   ├── raw/

│   └── processed/

├── models/

│   └── best\_next\_hour\_model.joblib

├── notebooks/

├── reports/

│   ├── figures/

│   └── tables/

├── src/

│   ├── 01\_inspect\_data.py

│   ├── 02\_deep\_data\_audit.py

│   ├── 03\_completeness\_audit.py

│   ├── 04\_build\_model\_dataset.py

│   ├── 05\_document\_route\_exclusion.py

│   ├── 06\_train\_evaluate\_models.py

│   ├── 07\_model\_error\_analysis.py

│   └── 08\_tram\_route\_diagnostics.py

└── requirements.txt

```



\### Technologies



\- Python 3.11

\- pandas

\- NumPy

\- scikit-learn

\- Matplotlib

\- Seaborn

\- joblib

\- Jupyter Notebook



\### Limitations



1\. QR validations do not represent all passengers.

2\. Nine calendar dates in the observed date range have no records.

3\. Historical data may not capture unexpected disruptions, special events or operational changes.

4\. The model's test performance does not establish that the same accuracy will continue in future periods.

5\. The route 1 peak-period analysis is based on a limited two-month test sample.

6\. Feature importance indicates model reliance, not causal relationships.



\### Reproducibility



Install the dependencies:



```bash

python -m pip install -r requirements.txt

```



Run the scripts in numerical order from the project root:



```bash

python src/01\_inspect\_data.py

python src/02\_deep\_data\_audit.py

python src/03\_completeness\_audit.py

python src/04\_build\_model\_dataset.py

python src/05\_document\_route\_exclusion.py

python src/06\_train\_evaluate\_models.py

python src/07\_model\_error\_analysis.py

python src/08\_tram\_route\_diagnostics.py

```



The original monthly CSV files must be placed in `data/raw/` before running the pipeline.



\### Research Purpose



This project demonstrates an applied machine learning workflow for public transport analytics, combining data quality assessment, time-series feature engineering, supervised regression, chronological evaluation, model explainability and critical error analysis.



It is an academic research prototype and not a deployed public transport scheduling system.



