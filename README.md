# GA-Optimized Neuro-Fuzzy System for Heatwave Risk Classification

Complete research prototype with synthetic demo, CSV training/inference, Streamlit interface, GA optimization, baseline comparisons, fuzzy rule exports and evaluation figures.

## Public demo

[Open Heatwave Risk Explorer](https://endangered-delivers-promoted-scientists.trycloudflare.com)

This temporary Cloudflare Tunnel link was created on 8 October 2026. It works only while the host Mac, local Streamlit app and tunnel remain running. It may be unavailable later, and restarting the tunnel creates a new URL. This is a shared local demo, not permanent cloud hosting.

The app defaults to New Delhi historical weather (2015–2025), with a synthetic demonstration available in the model selector. Risk categories are research-derived weather hazard labels, not official alerts or observed health outcomes.


## Quick start

Python 3.13 is recommended for the bundled trained models. Run from this folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-deploy.txt
python heatwave.py predict --model results_delhi/model.pkl --csv data/example_weather.csv
streamlit run app.py
python -m unittest discover -s tests -v
```

The repository includes trained historical and synthetic models; no retraining is required to launch the app. The default historical model is `results_delhi/model.pkl`; selecting the synthetic demo uses `results/model.pkl`. Run `python heatwave.py demo` to regenerate synthetic results or `python train_delhi.py` to retrain the historical model. These commands replace their respective result files. Only load trusted pickle files: pickle can execute code. Use the pinned deployment dependencies for the bundled models.

## Real CSV data

```sh
python heatwave.py train --csv data/labeled_weather.csv --out results_real --rules 6 --population 12 --generations 10
python heatwave.py predict --model results_real/model.pkl --csv data/example_weather.csv --out results_real/predictions.csv
```

| Column | Definition |
| --- | --- |
| max_temp_c | Daily maximum temperature in Celsius |
| min_temp_c | Daily minimum temperature in Celsius |
| relative_humidity_pct | Humidity, 0–100 percent |
| duration_days | Heat episode duration known at prediction time, at least 1 day |
| temp_anomaly_c | Anomaly against a documented historical local climatology in Celsius |
| risk | Training only: Low, Moderate, High, Extreme |

All four labels require at least 10 rows each. Missing, infinite and invalid values are rejected; extra columns are ignored. Record station, timestamp, humidity observation definition, label provenance and climatology outside this minimal schema. Never derive duration from future days or climatology from test observations.

## Method

Split observations into 60% training, 20% validation and 20% test, stratified by class. Fit StandardScaler and K-means only on training. Each fuzzy rule has a Gaussian center and width for every input. Product firing strengths are normalized in log space for numerical stability. Weighted first-order rule consequents produce class logits, followed by softmax. Logistic regression learns consequent coefficients with cross-entropy, class weighting and L2 regularization. This is a hybrid Takagi–Sugeno neuro-fuzzy classifier, rather than conventional scalar-output ANFIS regression.

GA chromosomes encode centers, log widths and log regularization. Tournament selection, arithmetic crossover, Gaussian mutation and elitism maximize validation macro-F1. The initial fixed-rule candidate is included. Freeze selected premises and the training-only scaler, refit consequents on train plus validation, and evaluate test once. Compare fixed-rule neuro-fuzzy, logistic regression and random forest on the same partitions. Baselines use illustrative defaults rather than equally tuned budgets.

For standardized input x, rule r has log firing `-0.5 * sum_j(((x_j-center_rj)/width_rj)^2)`. Normalize over rules. Class logit is `intercept_c + sum_r firing_r * (b_rc + sum_j a_rcj*x_j)`. Consequent coefficients are stored in the fitted model. Rule CSVs convert premises back to original units; centers do not establish a unique class or clinical rule.

Defaults: 6 rules, 12 individuals, 10 generations, tournament size 3, 2 elites, mutation probability .15 per gene and mutation scale .2. Bounds: centers [-4,4] standardized units, widths exp([-1.8,1.2]), C [.01,100]. Runtime scales with population times generations; improvement is not guaranteed.

## Outputs

- results/metrics.json: provenance, settings, split sizes, accuracy, macro-F1, per-class metrics and confusion matrices ordered Low, Moderate, High, Extreme.
- results/ga_history.csv: validation fitness by generation.
- results/evaluation.png: convergence and test confusion matrix.
- results/model.pkl: selected model, scaler and metadata.
- results/rule_centers.csv and rule_widths.csv: premises in input units.
- results/test_predictions.csv: held-out labels and predictions.
- results/synthetic_data.csv: invented demonstration observations.
- results/predictions.csv: sample batch predictions.

## Scientific scope

Synthetic label score is `.65*(Tmax-32)+.35*(Tmin-23)+.045*(humidity-45)+.42*duration+.4*anomaly+Normal(0,1.1)`. Thresholds 4,9,14 create four classes. These thresholds are invented, with no official or clinical meaning. Demo accuracy measures recovery of this artificial mechanism only. Weather inputs are hazard proxies and omit population vulnerability, exposure and cooling access. Scores are not calibrated probabilities of illness or death.

For real research, obtain authoritative observations and independently justified labels, document jurisdiction and climatology, and replace random splitting with chronological and station/event-grouped holdouts. CSV training still uses random splitting: it does not automatically prevent spatial or temporal leakage in correlated daily weather. Evaluate multiple seeds, regions and seasons, uncertainty/calibration, false alarms, minority-class recall and tuned baselines. Threshold-derived labels test imitation of thresholds, not independent health outcomes. This project does not provide official alerts or medical advice.

## References

WHO heat hazard and warning-system context; these do not validate the synthetic formula:
- https://www.who.int/news-room/fact-sheets/detail/climate-change-heat-and-health
- https://www.who.int/publications/m/item/heatwaves-and-health--guidance-on-warning-system-development

Split API:
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

## New Delhi historical dataset (added)

NASA POWER daily gridded meteorological estimates for 28.6139 N, 77.2090 E, 2015–2025: 4,018 complete records. These are real-world weather estimates, not station observations. The interface defaults to the historical model and retains a synthetic model selector.

```sh
python fetch_delhi.py
python train_delhi.py
streamlit run app.py
```

Downloads are cached in data/new_delhi/power_*.json. Provenance includes request URLs, source parameter definitions, timestamps and SHA-256 checksums. weather_1985_2025.csv retains all downloaded weather; delhi_2015_2025_labeled.csv contains study inputs and derived categories. Reference climatology is 1985–2014, with a circular +/-7 calendar-day window for means and percentiles. No 2015–2025 records enter reference statistics.

Four-class research mapping, in priority order:
- Extreme: Tmax >=47°C, OR Tmax >=40°C and anomaly >=6.5°C, OR Tmax >=40°C and local p97.5 exceedance.
- High: Tmax >=45°C, OR Tmax >=40°C and anomaly >=4.5°C.
- Moderate: Tmax >=38°C OR local p90 exceedance.
- Low: all remaining days.

Moderate and the percentile extension of Extreme are invented research definitions. High and part of Extreme use IMD-like daily temperature criteria; this implementation does not reproduce official IMD declarations, station/subdivision requirements, alert colors or health outcomes. Its locally calculated climatology is not an official IMD normal. See https://mausam.imd.gov.in/Forecast/marquee_data/Press%20Release%2017-02-2026.pdf for published criteria. Humidity is daily mean, not humidity concurrent with daily maximum temperature; no heat-index calculation is made.

Duration counts current/past consecutive days above local p90, with non-exceedance days mapped to 1 to satisfy the existing model schema. January 2015 initializes a fresh counter. Labels are deterministic functions of input temperature and anomaly: evaluation measures approximation of the documented categorization, not forecasting future heatwaves or predicting independent health risk.

Real-data training uses chronological partitions: 2015–2021 train, 2022–2023 validation, 2024–2025 test. All partitions must contain every class. The scaler is training-only. Consequents refit on train+validation after GA selection. Results are in results_delhi/. The generic train command supports --chronological with these fixed date boundaries; do not use its default random split for this dataset.

## Cloudflare sharing and hosting

The public demo uses Cloudflare Tunnel to forward HTTPS requests to the local Streamlit server. Cloudflare does not run the Python app in this setup. GitHub stores the source and artifacts; GitHub Pages cannot execute the Streamlit server.

To create a new temporary link, install `cloudflared` from [Cloudflare's official instructions](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/), start the app with the Quick start command, then run in a second terminal:

```sh
cloudflared tunnel --url http://127.0.0.1:8501 --no-autoupdate --protocol http2
```

Open the generated `https://...trycloudflare.com` URL. Keep both processes running and the host awake. Stop the tunnel with Ctrl+C to remove public access. No Cloudflare account is needed for a Quick Tunnel; it has no uptime guarantee. See [Cloudflare Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).

For persistent availability, run the app on an always-on server and configure a named Cloudflare Tunnel and hostname. A named tunnel still requires a running origin server. Alternatively, this repository includes `Dockerfile` and `render.yaml` for a future server deployment; no such deployment has been provisioned. Docker exposes port 8501. No API credentials are needed by the app.

Automated GitHub checks cover unit tests, bundled-model inference and the interface classification path. The initial ten project commits group components for publication rather than representing a backdated development timeline.
