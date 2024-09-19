import pandas as pd

# Load the CSV files into pandas DataFrames
manhattan_scaled_stats = pd.read_csv('output/manhattan_scaled_no_dd_updating_practice/stats.csv')
nn_mahalanobis_stats = pd.read_csv('output/nn_mahalanobis_no_dd_updating_practice/stats.csv')
outlier_count_stats = pd.read_csv('output/outlier_count_no_dd_updating_practice/stats.csv')
one_class_svm_stats = pd.read_csv('output/one_class_svm_no_dd_updating_practice/stats.csv')
mahalanobis_stats = pd.read_csv('output/mahalanobis_no_dd_updating_practice/stats.csv')

# Merge the DataFrames on 'trainset_size'
merged_df = pd.merge(manhattan_scaled_stats[['trainset_size', 'eer_avg']],
                     nn_mahalanobis_stats[['trainset_size', 'eer_avg']],
                     on='trainset_size', suffixes=('_manhattan_scaled', '_nn_mahalanobis'))

merged_df = pd.merge(merged_df, outlier_count_stats[['trainset_size', 'eer_avg']],
                     on='trainset_size')
merged_df = pd.merge(merged_df, one_class_svm_stats[['trainset_size', 'eer_avg']],
                     on='trainset_size', suffixes=('', '_one_class_svm'))
merged_df = pd.merge(merged_df, mahalanobis_stats[['trainset_size', 'eer_avg']],
                     on='trainset_size', suffixes=('', '_mahalanobis'))

# Rename columns for clarity
merged_df.columns = ['trainset_size', 'eer_avg_manhattan_scaled', 'eer_avg_nn_mahalanobis',
                     'eer_avg_outlier_count', 'eer_avg_one_class_svm', 'eer_avg_mahalanobis']

# Select only the columns needed for the TSV
final_df = merged_df[['eer_avg_manhattan_scaled', 'eer_avg_nn_mahalanobis',
                      'eer_avg_outlier_count', 'eer_avg_one_class_svm', 'eer_avg_mahalanobis']]

# Format the values to always show 3 decimal places
final_df = final_df.map(lambda x: f"{x:.3f}")

# Save the DataFrame to a TSV file
final_df.to_csv('output/combined_eer_avg_stats.tsv', sep='\t', index=False)
