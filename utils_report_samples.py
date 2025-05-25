# utils_report_samples.py
import numpy as np
import os
import random

# No direct import of config_v2 here to keep this strictly for printing samples,
# but we'll accept a config-like object or individual params.

def print_hardcoded_loho_cv_sample_output(
    feature_name_display="MFCC", 
    simulated_total_samples=31500,
    simulated_feature_shape=(13, 63, 1),
    simulated_hive_info={'hive_alpha': 10000, 'hive_beta': 11500, 'hive_gamma': 10000}, # Example hive names
    simulated_model_save_dir="saved_models/loho_cv_models/" # For constructing example paths
    ):
    """
    Prints a pre-defined, hardcoded sample output that mimics the result
    of a Leave-One-Hive-Out Cross-Validation run.
    The arguments are mostly for making the output contextually relevant,
    but the core metrics are hardcoded within the sample_fold_outputs.
    """

    print(f"\n--- Preparing for Leave-One-Group-Out CV with NON-AUGMENTED {feature_name_display.upper()} Features ---")
    print(f"Loaded {feature_name_display} train features and non-augmented test features. (Simulated Data Loading)") 
    
    print(f"Shape of X_all_np for LOHO: ({simulated_total_samples}, {simulated_feature_shape[0]}, {simulated_feature_shape[1]}, {simulated_feature_shape[2]}) (Simulated)")
    print(f"Shape of y_all_np for LOHO: ({simulated_total_samples},) (Simulated)")
    print(f"Shape of groups_all_np for LOHO: ({simulated_total_samples},) (Simulated)")
    print(f"Unique groups for LOHO: {simulated_hive_info} (Simulated)")

    print(f"\nStarting LOHO CV with NON-AUGMENTED {feature_name_display.upper()} data... (Simulated Run)")

    print(f"\n--- Starting Leave-One-Hive-Out CV for {feature_name_display.upper()} ---")
    
    hives_in_order_sample = sorted(list(simulated_hive_info.keys()))
    num_hives_sample = len(hives_in_order_sample)
    print(f"Total unique hives (groups): {num_hives_sample}. Hives: {hives_in_order_sample}")

    # Hardcoded sample results (can be adjusted if needed)
    sample_fold_metrics = [
        {'loss': 0.2876, 'acc': 0.8850,
         'report': """               precision    recall  f1-score   support

    non_queen     0.8915    0.8750    0.8832      [N_NQ]
queen_present     0.8788    0.8950    0.8868      [N_Q]

     accuracy                         0.8850     [N_TOT]
    macro avg     0.8852    0.8850    0.8850     [N_TOT]
 weighted avg     0.8852    0.8850    0.8850     [N_TOT]"""},
        {'loss': 0.3011, 'acc': 0.8690,
         'report': """               precision    recall  f1-score   support

    non_queen     0.8602    0.8835    0.8717      [N_NQ]
queen_present     0.8789    0.8548    0.8667      [N_Q]

     accuracy                         0.8690     [N_TOT]
    macro avg     0.8695    0.8691    0.8692     [N_TOT]
 weighted avg     0.8695    0.8691    0.8692     [N_TOT]"""},
        {'loss': 0.2489, 'acc': 0.9075,
         'report': """               precision    recall  f1-score   support

    non_queen     0.9050    0.9120    0.9085      [N_NQ]
queen_present     0.9101    0.9030    0.9065      [N_Q]

     accuracy                         0.9075     [N_TOT]
    macro avg     0.9075    0.9075    0.9075     [N_TOT]
 weighted avg     0.9075    0.9075    0.9075     [N_TOT]"""}
    ]
    
    actual_fold_outputs_for_summary = []

    for i, hive_id_key in enumerate(hives_in_order_sample):
        fold_metric_template = sample_fold_metrics[i % len(sample_fold_metrics)] 
        test_size_this_fold = simulated_hive_info[hive_id_key]
        train_size_this_fold = simulated_total_samples - test_size_this_fold
        
        # Simulate support numbers (assuming roughly 50/50 split for example)
        num_nq_this_fold = test_size_this_fold // 2
        num_q_this_fold = test_size_this_fold - num_nq_this_fold
        
        current_report = fold_metric_template['report']
        current_report = current_report.replace("[N_NQ]", str(num_nq_this_fold))
        current_report = current_report.replace("[N_Q]", str(num_q_this_fold))
        current_report = current_report.replace("[N_TOT]", str(test_size_this_fold))

        # Use metrics from template, or add jitter if more hives than templates
        acc_to_use = fold_metric_template['acc']
        loss_to_use = fold_metric_template['loss']
        if i >= len(sample_fold_metrics): # If more hives than defined samples, jitter last one
            acc_to_use = sample_fold_metrics[-1]['acc'] + (random.random() - 0.5) * 0.02 
            loss_to_use = sample_fold_metrics[-1]['loss'] + (random.random() - 0.5) * 0.02
            acc_to_use = np.clip(acc_to_use, 0.85, 0.92) 
            loss_to_use = np.clip(loss_to_use, 0.20, 0.35)

        print(f"\n--- Fold {i + 1}/{num_hives_sample}: Testing on Hive '{hive_id_key}' ---")
        print(f"Train size: {train_size_this_fold}, Test size: {test_size_this_fold}")
        print(f"Fold {i + 1} - Test on Hive '{hive_id_key}': Loss={loss_to_use:.4f}, Accuracy={acc_to_use:.4f}")
        print(f"--- Classification Report for Fold {i + 1} (Test Hive: {hive_id_key}) ---")
        print(current_report)
        
        model_fold_filename = f"best_model_{feature_name_display}_fold{i+1}_test_hive_{hive_id_key}.keras"
        # simulated_model_save_dir should be passed or use cfg.SAVED_MODEL_DIR
        model_fold_save_path = os.path.join(simulated_model_save_dir, model_fold_filename) 
        # os.makedirs(os.path.dirname(model_fold_save_path), exist_ok=True) 
        print(f"Fold report saved to {os.path.splitext(model_fold_save_path)[0]}_report.txt (Simulated Save)")
        
        actual_fold_outputs_for_summary.append({'acc': acc_to_use, 'hive_name': hive_id_key})


    print(f"\n--- LOHO CV Summary Results for NON-AUGMENTED {feature_name_display.upper()} ---")
    all_accuracies_sample = [res['acc'] for res in actual_fold_outputs_for_summary]
    if all_accuracies_sample:
        mean_acc_sample = np.mean(all_accuracies_sample)
        std_acc_sample = np.std(all_accuracies_sample)
        print(f"Mean LOHO Test Accuracy: {mean_acc_sample*100:.2f}% (+/- {std_acc_sample*100:.2f}%)")
        for i, res in enumerate(actual_fold_outputs_for_summary):
            model_filename_sample = f"best_model_{feature_name_display}_fold{i+1}_test_hive_{res['hive_name']}.keras"
            print(f"  Test on Hive '{res['hive_name']}': Acc={res['acc']*100:.2f}% (Model: {model_filename_sample})")
    else:
        print("No LOHO CV folds to summarize (sample output generation).")