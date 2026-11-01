# Assignment 3: AutoML on the athletes dataset

Predict **deadlift** (regression) from the CrossFit athletes data, using Azure ML Automated ML (the chosen MLOps platform) and H2O AutoML.

## Files

- `azureml_automl.ipynb`: Azure ML Automated ML (data prep, all features, top 5 features, top 3 features)
- `h2o_automl.ipynb`: H2O AutoML (data prep, all features, top 5 features, top 3 features)
- `environment.yml`: conda environment for the H2O notebook (Python, Java, pandas, h2o)
- `screenshots/azure/`: step-by-step screenshots of the Azure ML no-code (Studio) run, all features
- `screenshots/azure/top3/`: the same for the Studio run with the top 3 features
- `Azure Auto ML/` and `Azure Auto ML Top 3/`: files downloaded from those two Studio jobs (data guardrails result and job definition)
- `screenshots/h2o/` and `screenshots/h2o/top3/`: step-by-step screenshots of the H2O Flow (no-code) runs, all features and top 3 features
- `data/athletes.csv`: raw data (not committed)

## How to run

**H2O**
```
conda env create -f environment.yml
conda activate mlops2
```
Open `h2o_automl.ipynb` and run all cells.

**Azure ML**

Register the raw data as a data asset named `athletes_raw` (version 1), then open `azureml_automl.ipynb` on an Azure ML compute instance with the **Python 3.10 - SDK v2** kernel and run all cells. The jobs run on serverless compute.

## Data prep

Rows with missing key values are dropped, and columns that are not relevant (name, team, affiliate, region-level IDs and the benchmark workout columns) are removed. Outliers are removed (age under 18, height outside 48-96 in, weight over 1500, and lifts at or below 0 or above the limits in the notebooks). Survey answers marked "Decline to answer" are dropped.

Result: **30,029 rows, 14 columns**, with `deadlift` as the target.

## Azure ML Automated ML

### No-code run in Azure ML Studio

The same task was run in the Studio UI.

1. **Create the data asset:** Data > Create, name `athletes`, type Tabular, from local files, choose the `workspaceblobstore` datastore, upload `athletes_clean.csv`, check the schema, and create it.

   ![Data asset name and type](screenshots/azure/01-data-asset-name.png)
   ![Data source](screenshots/azure/02-data-source.png)
   ![Local files selected](screenshots/azure/03-local-files-selected.png)
   ![Select a datastore](screenshots/azure/04-select-datastore.png)
   ![Upload the file](screenshots/azure/05-upload-file.png)
   ![Schema](screenshots/azure/06-schema.png)
   ![Review and create](screenshots/azure/07-review-create.png)

2. **Submit the Automated ML job:** Automated ML > New job. Name it, choose task type Regression and the `athletes` asset, and set `deadlift` as the target.

   ![Basic settings](screenshots/azure/08-job-basic-settings.png)
   ![Task type and data](screenshots/azure/09-task-and-data.png)

3. **Limits and validation:** experiment timeout 15 minutes, 5-fold cross-validation and a 30% test split.

   ![Task settings](screenshots/azure/10-task-settings.png)

4. **Compute:** serverless, CPU, `Standard_DS3_v2`, 1 instance. Then review and submit.

   ![Compute](screenshots/azure/11-compute.png)
   ![Review and submit](screenshots/azure/12-review-submit.png)

5. **Result:** the job completed in about 27 minutes. The best model is a VotingEnsemble with a normalized RMSE of 0.04051. Azure ML stopped the experiment when the 15-minute timeout was reached, and the time for startup, featurization and the final steps came on top of that.

   ![Job completed](screenshots/azure/13-job-completed.png)

6. **Data insights (Data guardrails):** before training, Automated ML checks the data. The result is saved in `Azure Auto ML/outputs/verifier_results.json`.

   | Check | Result |
   |---|---|
   | Missing feature values imputation | Passed. No missing values were detected in the training data. |
   | High cardinality feature detection | Passed. No high cardinality features were detected. |

### No-code run with fewer features (Studio)

The assignment asks for a second run using only the top features, so the job was repeated in Studio on a reduced data asset, `athletes_top3`. Only the top 3 features from the notebook (`backsq`, `candj` and `gender`) and the `deadlift` target are included in its schema (see step 1).

1. **Create the reduced data asset:** Data > Create, name `athletes_top3`, type Tabular, from local files, `workspaceblobstore` datastore, upload `athletes_clean.csv`. On the Schema page, switch off every column except `backsq`, `candj`, `gender` and the `deadlift` target, then review and create.

   ![Data asset name](screenshots/azure/top3/01-data-asset-name.png)
   ![Local files source](screenshots/azure/top3/02-data-source.png)
   ![Select a datastore](screenshots/azure/top3/03-select-datastore.png)
   ![Upload the file](screenshots/azure/top3/04-upload-file.png)
   ![Parse settings and data preview](screenshots/azure/top3/05-parse-settings.png)
   ![Schema: only some columns included](screenshots/azure/top3/06-schema.png)
   ![Review and create](screenshots/azure/top3/07-review-create.png)
   ![athletes_top3 listed under Data assets](screenshots/azure/top3/08-data-asset-created.png)

2. **Submit the Automated ML job:** Automated ML > New job, name `athelete_top3`, task type Regression, data `athletes_top3`.

   ![Basic settings](screenshots/azure/top3/09-job-basic-settings.png)
   ![Task type and data](screenshots/azure/top3/10-task-and-data.png)

3. **Task settings:** target `deadlift`, experiment timeout 15 minutes, 5-fold cross-validation and a train-test split for the test data (same limits as the first run).

   ![Task settings](screenshots/azure/top3/11-task-settings.png)

4. **Compute and submit:** this run used the compute instance `salli1` (`Standard_DS11_v2`, 2 vCPUs) rather than serverless compute.

   ![Compute](screenshots/azure/top3/12-compute.png)
   ![Review and submit](screenshots/azure/top3/13-review-submit.png)

5. **Result:** the job completed in 19 min 4 s (duration 19m 3.630s). Azure ML stopped the experiment when the 15-minute timeout was reached, and startup, featurization and the final steps came on top of that.

   ![Job completed](screenshots/azure/top3/14-job-completed.png)

6. **Models:** the Models + child jobs tab lists the models by normalized RMSE. The best is a VotingEnsemble with a normalized RMSE of 0.04128.

   ![Models and child jobs](screenshots/azure/top3/15-models.png)

   | Rank | Model | Normalized RMSE | Duration |
   |---|---|---|---|
   | 1 | VotingEnsemble | 0.04128 | 1m 11s |
   | 2 | MaxAbsScaler, LightGBM | 0.04148 | 48s |
   | 3 | MaxAbsScaler, ExtremeRandomTrees | 0.04228 | 53s |

   By speed (shortest duration), the fastest models are StandardScalerWrapper ElasticNet (46s, 0.04256), MaxAbsScaler LightGBM (48s, 0.04148) and MaxAbsScaler RandomForest (49s, 0.04527). LightGBM is the best trade-off: second in score and second in speed.

### Notebook run (code)

Settings, matching the Studio run where Azure ML allows it: regression, primary metric normalized RMSE, 5-fold cross-validation, a 30% test split, all algorithms allowed, at most 20 trials (4 at a time), 14 minutes per trial and 60 minutes in total. The trial cap is what ends each job, so the two jobs do the same amount of search. Scores are RMSE in lbs from cross-validation.

**Top 5 features** (from the best model's explanation, job `neat_energy_tb9tnsb036`):

| Rank | Feature | Importance |
|---|---|---|
| 1 | backsq | 52.0 |
| 2 | candj | 17.2 |
| 3 | gender | 5.6 |
| 4 | weight | 4.3 |
| 5 | snatch | 2.6 |

**Top 3 models by validation score, all features** (job `neat_energy_tb9tnsb036`; the model names are Azure's auto-generated trial IDs):

| Rank | Trial | RMSE (lbs) | Training time |
|---|---|---|---|
| 1 | neat_energy_tb9tnsb036_18 | 36.18 | 56.0 s |
| 2 | neat_energy_tb9tnsb036_19 | 36.20 | 60.0 s |
| 3 | neat_energy_tb9tnsb036_0 | 36.39 | 34.2 s |

**Top 3 features (backsq, candj, gender), job `green_collar_29y0mszt1n`:** the best trial is `green_collar_29y0mszt1n_18` with an RMSE of 36.98 lbs and a training time of 52.9 s.

The leaderboard still searches the whole experiment, and the experiment names were reused, so the printed tables also contain trials from earlier jobs. The all-features table above keeps only the new job's trials, picked from the printed rows. An earlier job with the same settings but no trial cap (`jolly_camel_m6n2gt6mqs`) reached 36.18 too, so the score is stable across runs. The printed "top 3 by speed" tables and the second and third top-3-feature rows belong to older jobs (`great_chaconia`, `zen_drawer`, `olive_comb`), so they are not reported here.

**All features vs top 3 features (Part A is the all-features run)**

| Run | RMSE (lbs) | Best model's training time | Total job time |
|---|---|---|---|
| All features (Part A) | 36.18 | 56 s | 41.3 min |
| Top 3 features (backsq, candj, gender) | 36.98 | 53 s | 23.3 min |

The top-3 model is about 0.80 lbs worse (2.2%), because `backsq` carries most of the signal and `candj` adds most of the rest. The best model trains in about the same time (56 s against 53 s). The top-3 job finishes in about half the time (23.3 against 41.3 minutes), because both jobs stop after 20 trials and each trial is cheaper with 3 columns, and the startup and finishing steps vary by several minutes. Azure ML does not report prediction time per row for AutoML models, so "speed" here is training time.

### Is Azure ML Automated ML no-code, low-code or full-code?

**No-code in Azure ML Studio, low-code in the notebook.** In Studio, the data, target and limits are chosen in a form with no code. In the notebook, the same steps are one `automl.regression(...)` call plus a few settings (trial limit, time limit, allowed algorithms), and Azure ML chooses the preprocessing, models and tuning itself.

## H2O AutoML

### No-code run in H2O Flow

The same run was repeated in H2O Flow, the browser UI at `http://localhost:54321`, with no code.

1. **Upload the cleaned data:** Data > Upload File, then choose `athletes_clean.csv`.

   ![Data menu with Upload File](screenshots/h2o/01-upload-menu.png)
   ![Choose file dialog](screenshots/h2o/02-choose-file.png)

2. **Parse it:** check the column names and types (numbers for the lifts, Enum for the text columns), then click Parse. The parse finishes with 14 columns.

   ![Parse setup](screenshots/h2o/03-parse-setup.png)
   ![Parse button](screenshots/h2o/04-parse-button.png)
   ![Parse done](screenshots/h2o/05-parse-done.png)

3. **Split 80/20:** Data > Split Frame, with ratios 0.80 and 0.20. This gives 24,005 training rows and 6,024 validation rows.

   ![Split Frame menu](screenshots/h2o/06-split-menu.png)
   ![Split Frame settings](screenshots/h2o/07-split-frame.png)

4. **Run AutoML:** Model > Run AutoML. Set the training frame to the 0.80 split, the response column to `deadlift` and the validation frame to the 0.20 split. No columns are ignored, so all 13 features are used. DeepLearning is excluded and `max_models` is 15. Then click Build Models.

   ![Run AutoML menu](screenshots/h2o/08-run-automl-menu.png)
   ![Frames and response column](screenshots/h2o/09-automl-frames.png)
   ![Excluded algorithm and max models](screenshots/h2o/10-automl-algos.png)
   ![Build Models](screenshots/h2o/11-automl-build.png)

5. **Read the results:** the leaderboard and the event log.

   ![Leaderboard and event log](screenshots/h2o/12-results.png)

**UI results (all features):** 17 models were built, ranked by RMSE in lbs. Flow's split and AutoML seed (random) differ from the notebook run, so these numbers are close to, but not the same as, the notebook's.

| Rank | Model | RMSE | MAE |
|---|---|---|---|
| 1 | Stacked Ensemble AllModels | 36.24 | 27.29 |
| 2 | Stacked Ensemble BestOfFamily | 36.29 | 27.33 |
| 3 | GBM_grid_1 model 4 | 36.63 | 27.61 |

### No-code run with fewer features (Flow)

The same steps were repeated in Flow with only 3 features (`backsq`, `candj`, `snatch`). The other 10 columns were ticked in `ignored_columns`.

1. **Upload and parse:** Data > Upload File, `athletes_clean.csv`, then Parse (14 columns).

   ![Upload File menu](screenshots/h2o/top3/01-upload-menu.png)
   ![Choose file dialog](screenshots/h2o/top3/02-choose-file.png)
   ![Parse setup](screenshots/h2o/top3/03-parse-setup.png)
   ![Parse done](screenshots/h2o/top3/04-parse-done.png)

2. **Split 80/20:** Data > Split Frame, ratios 0.80 and 0.20.

   ![Split Frame menu](screenshots/h2o/top3/05-split-menu.png)
   ![Split Frame settings](screenshots/h2o/top3/06-split-frame.png)
   ![Split result](screenshots/h2o/top3/07-split-done.png)

3. **Run AutoML:** Model > Run AutoML, response column `deadlift`, region, gender, age, height, weight, eat and the other text columns ignored, DeepLearning excluded, `max_models` 15.

   ![Run AutoML menu](screenshots/h2o/top3/08-run-automl-menu.png)
   ![Frames and response column](screenshots/h2o/top3/09-automl-frames.png)
   ![Ignored columns, excluded algorithm and max models](screenshots/h2o/top3/10-automl-columns-algos.png)
   ![Build Models](screenshots/h2o/top3/11-automl-build.png)

4. **Read the results:** the leaderboard and the event log (project `athletes_top3`).

   ![Leaderboard and event log](screenshots/h2o/top3/12-results.png)

**UI results (top 3 features):** the leaderboard lists two AutoML runs under the same project name. The final run (`AutoML_4`) built 17 models, and its top 3 are below. The rows with RMSE of about 54 are from an earlier run (`AutoML_3`) and are not used.

| Rank | Model | RMSE | MAE |
|---|---|---|---|
| 1 | Stacked Ensemble AllModels | 37.30 | 27.78 |
| 2 | Stacked Ensemble BestOfFamily | 38.28 | 28.69 |
| 3 | GBM_10 | 38.32 | 28.69 |

The `runAutoML` request confirms the settings: training frame `_0.80`, validation frame `_0.20`, response `deadlift`, the 10 other columns ignored, DeepLearning excluded and `max_models` 15. The job took 17.8 seconds.

Compared with the Flow all-features run (36.24), using only 3 features is about 1.1 lbs worse (3.0%).

### Notebook run (code)

Settings: up to 15 models, seed 42, Deep Learning excluded, 5-fold cross-validation (the H2O default). Scores are RMSE in lbs.

**Data insights from the AutoML run**

- AutoML used 5-fold cross-validation. The validation frame only gave extra, informational metrics, so the leaderboard RMSE is a cross-validation score.
- XGBoost was not available on this machine and was skipped.
- It built 17 models in 56 seconds (H2O-reported; 110.5 s wall-clock in Python): 12 GBM, 1 GLM, 1 XRT, 1 DRF and 2 Stacked Ensembles.
- RMSE ranged from 36.6 (best Stacked Ensemble) to 40.9 (worst GBM grid model). The best single model was `GBM_5` at 37.0, so the ensembles gain only about 0.4 lbs.

**Top 5 features.** The best model is a Stacked Ensemble, which has no feature importance, so the importance comes from the best single model, `GBM_5`.

| Rank | Feature | Importance |
|---|---|---|
| 1 | backsq | 70.1% |
| 2 | candj | 21.0% |
| 3 | snatch | 2.7% |
| 4 | weight | 1.5% |
| 5 | schedule | 1.1% |

**Top 3 models by validation score**

| Rank | All features | Top 3 features (backsq, candj, snatch) |
|---|---|---|
| 1 | Stacked Ensemble AllModels, 36.60 | Stacked Ensemble AllModels, 37.99 |
| 2 | Stacked Ensemble BestOfFamily, 36.63 | Stacked Ensemble BestOfFamily, 38.01 |
| 3 | GBM_5, 37.00 | GBM_5, 38.05 |

The same three models win with both feature sets. Using only 3 of 13 features raises the best RMSE by about 1.4 lbs (3.8%), because `backsq` and `candj` carry most of the signal.

**Comparison with Part A (all features).** The all-features run is the Part A model, so the top-3 run is compared with it. The top-3 run is 1.4 lbs worse in validation score, with the same three models winning. It is faster to build: the AutoML run took 16 s against 56 s (H2O-reported; 72.8 s against 110.5 s wall-clock in Python, which includes sending data and fetching the leaderboard). The best model also trains and predicts a little faster with fewer features:

| Model | RMSE | Training time (ms) | Prediction time per row (ms) |
|---|---|---|---|
| All features, Stacked Ensemble AllModels | 36.60 | 1165 | 0.0168 |
| All features, Stacked Ensemble BestOfFamily | 36.63 | 443 | 0.0059 |
| All features, GBM_5 | 37.00 | 255 | 0.0010 |
| Top 3 features, Stacked Ensemble AllModels | 37.99 | 1094 | 0.0098 |
| Top 3 features, Stacked Ensemble BestOfFamily | 38.01 | 411 | 0.0037 |
| Top 3 features, GBM_5 | 38.05 | 134 | 0.0009 |

The best model is the slowest to predict, since it blends many models. `GBM_5` is about 16 times faster to predict than the ensemble on all features, for 0.4 lbs more error.

### Is H2O AutoML no-code, low-code or full-code?

**No-code in H2O Flow, low-code in the notebook.** In Flow, the whole run (upload, parse, split, AutoML settings, leaderboard) is done with menus and forms and no code. In the notebook, the modelling is one `H2OAutoML(...)` object with a few settings (`max_models`, `seed`, `exclude_algos`) and one `train(...)` call. H2O chooses the algorithms, tuning, cross-validation and ensembles itself. It is not full-code because no model is coded or tuned by hand. The notebook still needs Python and Java to be set up (see `environment.yml`), which Azure ML Studio does not.

## Both platforms

**Summary of the eight runs** (best model per run)

| Platform | Run | Features | Best model | Validation RMSE (lbs) | Time |
|---|---|---|---|---|---|
| Azure ML | Studio (no-code) | All | VotingEnsemble | 0.04051 (normalized) | 27 min 2 s |
| Azure ML | Studio (no-code) | Top 3 | VotingEnsemble | 0.04128 (normalized) | 19 min 4 s |
| Azure ML | Notebook (low-code) | All | trial `neat_energy_tb9tnsb036_18` | 36.18 | 41.3 min |
| Azure ML | Notebook (low-code) | Top 3 | trial `green_collar_29y0mszt1n_18` | 36.98 | 23.3 min |
| H2O | Flow (no-code) | All | Stacked Ensemble AllModels | 36.24 | 46 s |
| H2O | Flow (no-code) | Top 3 | Stacked Ensemble AllModels | 37.30 | 18 s |
| H2O | Notebook (low-code) | All | Stacked Ensemble AllModels | 36.60 | 56 s (110.5 s wall-clock) |
| H2O | Notebook (low-code) | Top 3 | Stacked Ensemble AllModels | 37.99 | 16 s (72.8 s wall-clock) |

- The top two features (`backsq` and `candj`) agree. Lower ranks differ because the two platforms use different importance methods.
- Azure ML's top 3 features are `backsq`, `candj` and `gender`, and H2O's are `backsq`, `candj` and `snatch`, so the two top-3-feature runs are not identical.
- Azure ML returns a single best model per job and reports training time per trial but not prediction speed. H2O gives a full leaderboard.

## Summary of the results

- **Accuracy.** With all features, the three runs scored in lbs land within 0.4 lbs of each other: Azure ML notebook 36.18, H2O Flow 36.24 and H2O notebook 36.60. Azure ML Studio reports a normalized RMSE (0.04051), which is not directly comparable. Dropping to the top 3 features costs 0.8 to 1.4 lbs (2% to 4%) on every platform, so three lifts give about 96% to 98% of the full model's accuracy.
- **Features.** `backsq` is the strongest predictor on both platforms (52% on Azure ML, 70% on H2O) and `candj` is second. The rest add little, which is why the top-3 models are only slightly worse.
- **Speed.** H2O finished every run in under a minute (16 to 56 s), while Azure ML took 19 to 41 minutes. Azure ML's time is mostly startup, featurization and a time or trial limit on a cloud cluster, not model training: its best models train in about 50 to 60 s. Fewer features made every job shorter: H2O 56 s to 16 s, Azure ML Studio 27 to 19 min and Azure ML notebook 41 to 23 min.
- **Models.** The Studio and H2O winners are ensembles (VotingEnsemble, Stacked Ensembles). In H2O the ensemble beats the best single model (`GBM_5`) by only about 0.4 lbs and is about 16 times slower to predict, so a single GBM is the practical choice when prediction speed matters.
- **No-code vs low-code.** The no-code (Studio, Flow) and low-code (notebook) runs reach nearly the same scores. The notebooks add reproducibility and timing control, and the UIs need no setup.
