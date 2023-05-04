import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


for detector_name in (
    'euclidean',
    'nn_mahalanobis',
    'mahalanobis',
    'manhattan',
    'manhattan_scaled',
    'one_class_svm',
    'outlier_count'
):
    for exclude_dd_feature in (True, False):
        detector_dir_path = f'output/{detector_name}{"_no_dd" * exclude_dd_feature}'
        stats_file_path = f'{detector_dir_path}/stats.csv'

        try:
            stats = pd.read_csv(stats_file_path)
        except FileNotFoundError:
            print(f"{stats_file_path} not found")
            continue

        fig, ax = plt.subplots()
        fig.set_size_inches(2 * 6.4, 4.8)
        ax.plot(stats['trainset_size'], stats['eer_avg'])
        ax.set_xticks(np.arange(0, 210, 10))
        ax.set_yticks(np.arange(0, 0.5, 0.05))
        ax.grid(True)
        eer_avg_file_path = f'{detector_dir_path}/eer_avg_over_trainsize.png'
        plt.savefig(eer_avg_file_path)
        plt.close()
