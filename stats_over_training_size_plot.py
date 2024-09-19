import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

manhattan_scaled_stats = pd.read_csv('output/manhattan_scaled_no_dd_updating_practice/stats.csv')
nn_mahalanobis_stats = pd.read_csv('output/nn_mahalanobis_no_dd_updating_practice/stats.csv')
outlier_count_stats = pd.read_csv('output/outlier_count_no_dd_updating_practice/stats.csv')
one_class_svm_stats = pd.read_csv('output/one_class_svm_no_dd_updating_practice/stats.csv')
mahalanobis_stats = pd.read_csv('output/mahalanobis_no_dd_updating_practice/stats.csv')

plt.rcParams.update({'font.size': 14})

fig, ax = plt.subplots()
fig.set_size_inches(2 * 6.4, 8)
ax.plot(manhattan_scaled_stats['trainset_size'], manhattan_scaled_stats['eer_avg'], 'bv-', linewidth=1, markerfacecolor='none', label='Manhattan (scaled)')
ax.plot(nn_mahalanobis_stats['trainset_size'], nn_mahalanobis_stats['eer_avg'], 'go-', linewidth=1, markerfacecolor='none', label='NN mahalanobis')
ax.plot(outlier_count_stats['trainset_size'], outlier_count_stats['eer_avg'], 'r+-', linewidth=1, label='Outlier count')
ax.plot(one_class_svm_stats['trainset_size'], one_class_svm_stats['eer_avg'], 'x-', label='One-class SVM')
ax.plot(mahalanobis_stats['trainset_size'], mahalanobis_stats['eer_avg'], 'o-', label='Mahalanobis')
ax.set_xticks(np.arange(0, 110, 10))
ax.set_xlabel('Training set size')
ax.set_yticks(np.arange(0, 0.4, 0.05))
ax.set_ylabel('EER')
ax.grid(True)
ax.legend()
plt.savefig('output/stats.png')
plt.close()
