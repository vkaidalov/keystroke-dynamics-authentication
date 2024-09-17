#python evaluation.py --detector manhattan_scaled --exclude_ud_feature True --exclude_enter_feature True --use_sliding_window True # EER=7.1%
#python evaluation.py --detector manhattan_scaled --exclude_ud_feature True --exclude_enter_feature True --use_sliding_window True --impostors_practice True  # EER=9.7%

python evaluation.py --detector nn_mahalanobis --exclude_dd_feature True --use_sliding_window True --impostors_practice True
python evaluation.py --detector manhattan_scaled --exclude_dd_feature True --use_sliding_window True --impostors_practice True
python evaluation.py --detector outlier_count --exclude_dd_feature True --use_sliding_window True --impostors_practice True
python evaluation.py --detector mahalanobis --exclude_dd_feature True --use_sliding_window True --impostors_practice True
python evaluation.py --detector one_class_svm --exclude_dd_feature True --use_sliding_window True --impostors_practice True
