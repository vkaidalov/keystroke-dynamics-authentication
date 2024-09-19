import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

samples = pd.read_csv('DSL-StrongPasswordData.csv')

subject_ids: np.ndarray = samples['subject'].unique()

roc_curve_data_dir_path = 'output/outlier_count_no_dd_updating_practice/train_15'

for subject_id in subject_ids:
    roc_curve_data_file_path = f'{roc_curve_data_dir_path}/roc_{subject_id}.pkl'
    with open(roc_curve_data_file_path, 'rb') as f:
        (
            true_positive_rate,
            false_positive_rate,
            zmfar_idx,
            equal_error_rate_idx
        ) = pickle.load(f)

    print(subject_id)
    print(f'EER: {sorted([round(1 - true_positive_rate[equal_error_rate_idx], 3), false_positive_rate[equal_error_rate_idx]])}')
    print(f'ZMFAR: {false_positive_rate[zmfar_idx]}')

    plt.figure()
    plt.plot(false_positive_rate, true_positive_rate, 'x-')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("False Positive Rate (False Alarm Rate)")
    plt.ylabel("True Positive Rate (Hit Rate)")

    plt.plot(
        false_positive_rate[zmfar_idx],
        true_positive_rate[zmfar_idx],
        'o', color='black'
    )
    plt.text(
        false_positive_rate[zmfar_idx],
        true_positive_rate[zmfar_idx] - 0.075,
        'Zero-Miss False Alarm Rate',
        fontsize=12
    )
    plt.plot(
        false_positive_rate[equal_error_rate_idx],
        true_positive_rate[equal_error_rate_idx],
        'o', color='black'
    )
    plt.text(
        false_positive_rate[equal_error_rate_idx],
        true_positive_rate[equal_error_rate_idx] - 0.075,
        'Equal Error Rate',
        fontsize=12
    )

    roc_curve_img_file_path = roc_curve_data_file_path.replace('.pkl', '.png')
    plt.savefig(roc_curve_img_file_path)
    plt.close()
