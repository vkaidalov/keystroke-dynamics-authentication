# Keystroke Dynamics Authentication

This project evaluates keystroke dynamics as a behavioral biometric for password-based authentication. Rather than
relying only on whether a password is correct, it models how a person types it: key hold times (`H`), key-down to
key-down intervals (`DD`), and key-up to key-down intervals (`UD`). A typing sample that differs sufficiently from the
enrolled user's pattern is treated as anomalous (potentially an impostor).

The code accompanies the paper [*Improving Keystroke Dynamics Authentication: Balancing Accuracy and User Experience
Through Efficient Training*](https://doi.org/10.32782/IT/2024-3-5) (https://doi.org/10.32782/IT/2024-3-5). 
A local copy is available as [`keystroke-dynamics-auth-article.pdf`](keystroke-dynamics-auth-article.pdf).

## Objectives

- Replicate and extend the Killourhy and Maxion evaluation protocol using the public DSL-StrongPassword dataset.
- Compare anomaly detectors and study the trade-off between authentication accuracy and the number of password
  repetitions required for enrolment.
- Support both a fixed enrolment model and an updating (sliding-window) model that simulates periodic retraining.

## Results

The study confirms that keystroke dynamics can strengthen two-factor authentication while keeping enrolment practical.
Its key finding is that the **scaled Manhattan** and **Outlier Count (z-score)** detectors perform relatively well with
small training sets; this is especially useful when users are unwilling to type many password repetitions during
enrolment.

Performance is evaluated per subject using:

- **Equal Error Rate (EER):** the operating point where false-alarm and miss rates are equal; lower is better.
- **Zero-Miss False-Alarm Rate (ZMFAR):** the lowest false-alarm rate achievable with no missed impostors; lower is
  better.
- **ROC curves** and per-subject EER/ZMFAR distributions.

The results should be interpreted as research evaluation results, not as a production authentication policy or a fixed
universal threshold.

## Repository layout

| Path                                                                  | Purpose                                                                                                                                                              |
|-----------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DSL-StrongPasswordData.csv`                                          | Included benchmark dataset. The first three columns identify the subject/session/repetition; remaining columns are timing features.                                  |
| `anomaly_detector/`                                                   | Common detector interface and implementations: Euclidean, Manhattan, scaled Manhattan, Mahalanobis, nearest-neighbour Mahalanobis, One-Class SVM, and Outlier Count. |
| `evaluation.py`                                                       | Sequential experiment runner for training sizes 5 through 100.                                                                                                       |
| `evaluation_parallelised.py`                                          | Process-pool runner for training sizes 40 through 100 (up to six workers).                                                                                           |
| `plot_roc.py`, `plot_single_roc.py`, `plot_eer_zmfar_hist.py`         | Create ROC charts and EER/ZMFAR histograms from saved experiment data.                                                                                               |
| `stats_over_training_size_plot.py`, `stats_over_training_size_tsv.py` | Compare detector EERs across training-set sizes.                                                                                                                     |
| `app.py`, `app_entrypoint.sh`                                         | Interactive Voila demonstration application.                                                                                                                         |

## Setup

The supplied Conda environment targets Python 3.14:

```bash
conda env create -f environment.yml
conda activate keystroke-dynamics-authentication
```

Run every command below from the repository root, where the CSV dataset is located.

## Running an evaluation

`evaluation.py` trains one detector per subject and scores genuine-user and impostor samples. It writes all generated
artifacts under `output/<configuration>/`:

- `train_<size>/roc_<subject>.pkl` - ROC data and the EER/ZMFAR indices for one subject.
- `train_<size>/eer_hist.pkl` and `zmfar_hist.pkl` - per-subject metric arrays.
- `stats.csv` - average and standard deviation of EER and ZMFAR for each training size.

Choose one of these detector identifiers:

```text
euclidean, manhattan, manhattan_scaled, mahalanobis,
nn_mahalanobis, one_class_svm, outlier_count
```

For example, this reproduces the updating, impostor-practice configuration used by the comparison scripts while
excluding `DD` features:

```bash
python evaluation.py --detector manhattan_scaled --exclude_dd_features True --use_sliding_window True --impostors_practice True
```

Available switches are:

| Switch                          | Effect                                                                                                |
|---------------------------------|-------------------------------------------------------------------------------------------------------|
| `--detector NAME`               | Detector identifier; defaults to `manhattan`.                                                         |
| `--exclude_dd_features True`    | Drop key-down/key-down interval features.                                                             |
| `--exclude_ud_features True`    | Drop key-up/key-down interval features.                                                               |
| `--exclude_enter_features True` | Drop features involving the Return key.                                                               |
| `--use_sliding_window True`     | Re-train on a moving window and score the following five genuine samples at each step.                |
| `--impostors_practice True`     | Use the last five repetitions from other subjects as impostor samples; omit it to use the first five. |

Omit a switch to leave it disabled. These scripts use Python's `bool` argument conversion, so use the literal `True` to
enable an option; do not pass `False` expecting it to disable one.

For parallel processing of training sizes 40-100, use:

```bash
python evaluation_parallelised.py --detector outlier_count --exclude_dd_features True --use_sliding_window True --impostors_practice True
```

The parallel version starts up to six worker processes. Reduce `max_workers` in `evaluation_parallelised.py` if the
machine has limited memory or CPU resources.

## Creating plots and comparison tables

Run plotting scripts after the corresponding evaluation has produced its `output/` directory:

```bash
python plot_roc.py --detector manhattan_scaled --exclude_dd_feature True
python plot_eer_zmfar_hist.py --detector manhattan_scaled
python plot_single_roc.py
python stats_over_training_size_plot.py
python stats_over_training_size_tsv.py
```

`plot_roc.py` produces a ROC PNG for each subject and training size. `plot_eer_zmfar_hist.py` creates EER and ZMFAR
histogram PNGs. `plot_single_roc.py` and the two `stats_over_training_size_*` scripts contain a configuration path /
detector list near the top of the file; adjust those values to match the experiment you ran before executing them. The
comparison plot is saved as `output/stats.png`, and the comparison table as `output/combined_eer_avg_stats.tsv`.

Some plotting defaults reflect the experiments used for the paper rather than the current evaluator defaults. Before
plotting, make the `training_data_size` lists match the sizes you generated. In particular, `plot_eer_zmfar_hist.py`
does not incorporate its `--exclude_dd_feature` value into its input directory, so edit `detector_dir_path` there when
plotting a suffixed evaluation directory such as `manhattan_scaled_no_dd_updating_practice`.

## Interactive Voila app

The Voila app is a demonstration of enrollment, anomaly scoring, and optional retraining - it is not a production-ready
login service. It exposes 20 sample profiles and their passwords from the benchmark data.

Start it with:

```bash
bash app_entrypoint.sh
```

On Windows without Bash, run the command from `app_entrypoint.sh` directly, for example:

```bash
voila app.py --Voila.log_level=20 --VoilaConfiguration.show_tracebacks=True --autoreload=True --no-browser --enable_nbextensions=True --VoilaConfiguration.extension_language_mapping='{\".py\": \"python\"}'
```

It serves the app at <http://localhost:8866> by default.

In the app:

1. Select a **User Profile**. The displayed password hint identifies the sample password for that profile.
2. Click the password field and type the password. The app records browser keydown/keyup events and derives hold (`H`)
   and up-down (`UD`) timings.
3. Press **Enter** to complete the attempt. A matching password is scored by every detector that has previously been
   trained for that profile.
4. Leave **Re-train detectors after scoring** enabled to add the successful sample to each profile's enrolment data and
   retrain. The first successful entry establishes the initial model, so its score is `0`.
5. Inspect **Entered Keys** and **Scores**. The score table records the latest values first and whether retraining
   occurred.
6. Select **Save everything to file** to persist profiles, detector state, training samples, and score history to
   `app_state.pkl`. On a later launch the app restores that file automatically. Delete or move it to return to the
   built-in initial profiles.

The entry widget colours a prefix mismatch red and does not score an incorrect password. A correct password with very
different timing may still receive a high anomaly score; the demonstration records scores but does not apply a threshold
to deny the correct password.

## License

See [LICENSE](LICENSE).
