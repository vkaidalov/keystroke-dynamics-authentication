import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# for detector_name in (
#     'euclidean',
#     'nn_mahalanobis',
#     'mahalanobis',
#     'manhattan',
#     'manhattan_scaled',
#     'one_class_svm',
#     'outlier_count'
# ):
#     for exclude_dd_feature in (True, False):
#         detector_dir_path = f'output/{detector_name}{"_no_dd" * exclude_dd_feature}'
#         stats_file_path = f'{detector_dir_path}/stats.csv'
#
#         try:
#             stats = pd.read_csv(stats_file_path)
#         except FileNotFoundError:
#             print(f"{stats_file_path} not found")
#             continue
#
#         fig, ax = plt.subplots()
#         fig.set_size_inches(2 * 6.4, 4.8)
#         ax.plot(stats['trainset_size'], stats['eer_avg'])
#         ax.set_xticks(np.arange(0, 210, 10))
#         ax.set_xlabel('Розмір тренувального набору даних')
#         ax.set_yticks(np.arange(0, 0.5, 0.05))
#         ax.set_ylabel('EER')
#         ax.grid(True)
#         eer_avg_file_path = f'{detector_dir_path}/eer_avg_over_trainsize.png'
#         plt.savefig(eer_avg_file_path)
#         plt.close()

manh_and_eucl_file_path = 'output/manh_and_eucl.png'
svm_file_path = 'output/svm.png'
maha_both_file_path = 'output/mahalanobis_both.png'
manh_sc_and_out_c_file_path = 'output/manh_sc_outlier.png'
manh_sc_wo_dd_file_path = 'output/manh_sc_wo_dd.png'
outlier_wo_dd_file_path = 'output/outlier_wo_dd.png'
eucl_stats = pd.read_csv('output/euclidean/stats.csv')
manh_stats = pd.read_csv('output/manhattan/stats.csv')
svm_stats = pd.read_csv('output/one_class_svm/stats.csv')
maha_stats = pd.read_csv('output/mahalanobis/stats.csv')
nn_maha_stats = pd.read_csv('output/nn_mahalanobis/stats.csv')
manh_sc_stats = pd.read_csv('output/manhattan_scaled/stats.csv')
manh_sc_no_dd_stats = pd.read_csv('output/manhattan_scaled_no_dd/stats.csv')
outlier_c_stats = pd.read_csv('output/outlier_count/stats.csv')
outlier_no_dd_stats = pd.read_csv('output/outlier_count_no_dd/stats.csv')

fig, ax = plt.subplots()
fig.set_size_inches(2 * 6.4, 4.8)
# ax.plot(eucl_stats['trainset_size'], eucl_stats['eer_avg'], label='Евклідова відстань')
# ax.plot(manh_stats['trainset_size'], manh_stats['eer_avg'], label='Мангеттенська відстань')
# ax.plot(svm_stats['trainset_size'], svm_stats['eer_avg'], label='Однокласовий SVM')
# ax.plot(maha_stats['trainset_size'], maha_stats['eer_avg'], label='Махаланобіс')
# ax.plot(nn_maha_stats['trainset_size'], nn_maha_stats['eer_avg'], label='Махаланобіс-НС')
# ax.plot(manh_sc_stats['trainset_size'], manh_sc_stats['eer_avg'], label='Масштабована Мангеттенська')
# ax.plot(manh_sc_no_dd_stats['trainset_size'], manh_sc_no_dd_stats['eer_avg'], label='Масштабована Мангеттенська, без DD')
ax.plot(outlier_c_stats['trainset_size'], outlier_c_stats['eer_avg'], label='Лічильник за z-оцінкою')
ax.plot(outlier_no_dd_stats['trainset_size'], outlier_no_dd_stats['eer_avg'], label='Лічильник за z-оцінкою, без DD')
ax.set_xticks(np.arange(0, 210, 10))
ax.set_xlabel('Розмір тренувального набору даних')
ax.set_yticks(np.arange(0, 0.5, 0.05))
ax.set_ylabel('EER')
ax.grid(True)
ax.legend()
plt.savefig(outlier_wo_dd_file_path)
plt.close()
