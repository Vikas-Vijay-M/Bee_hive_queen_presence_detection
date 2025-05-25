# utils_report_samples.py
import numpy as np
import os
import random # Make sure this is imported if you use the jitter for more than 3 hives

def print_hardcoded_loho_cv_sample_output(
    feature_name_display="MFCC", 
    simulated_total_samples=31500,
    simulated_feature_shape=(13, 63, 1),
    simulated_hive_info={'hive_alpha': 10000, 'hive_beta': 11500, 'hive_gamma': 10000},
    simulated_model_save_dir="saved_models/loho_cv_models/"
    ):
    """
    Prints a pre-defined, hardcoded sample output that mimics the result
    of a Leave-One-Hive-Out Cross-Validation run for a GIVEN FEATURE TYPE.
    """

    print(f"\n\n==================================================================================")
    print(f"===== SAMPLE LOHO CV OUTPUT FOR: {feature_name_display.upper()} FEATURES =====")
    print(f"==================================================================================")


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

    # Slightly different sample metrics for MFCC vs MelSpec for illustration
    if feature_name_display.upper() == "MFCC":
        base_acc = [0.8850, 0.8690, 0.9075]
        base_loss = [0.2876, 0.3011, 0.2489]
    elif feature_name_display.upper() == "MELSPEC":
        base_acc = [0.9020, 0.8850, 0.9150] # Assuming MelSpec might do slightly better
        base_loss = [0.2550, 0.2800, 0.2200]
    else: # Default if other feature type
        base_acc = [0.8700, 0.8500, 0.8900]
        base_loss = [0.3000, 0.3200, 0.2700]

    sample_fold_metrics_template = [
        {'loss': base_loss[0], 'acc': base_acc[0],
         'report': """               precision    recall  f1-score   support

    non_queen     {nq_p:.4f}    {nq_r:.4f}    {nq_f1:.4f}      [N_NQ]
queen_present     {q_p:.4f}    {q_r:.4f}    {q_f1:.4f}      [N_Q]

     accuracy                         {acc:.4f}     [N_TOT]
    macro avg     {macro_p:.4f}    {macro_r:.4f}    {macro_f1:.4f}     [N_TOT]
 weighted avg     {w_avg_p:.4f}    {w_avg_r:.4f}    {w_avg_f1:.4f}     [N_TOT]"""},
        # Add more templates if you want more variation for >3 hives, or it will cycle
    ]
    # Use more distinct metrics for the 3 sample folds
    if feature_name_display.upper() == "MFCC":
        detailed_metrics_per_fold = [ # P_nq, R_nq, P_q, R_q, Acc
            [0.8915, 0.8750, 0.8788, 0.8950, base_acc[0]],
            [0.8602, 0.8835, 0.8789, 0.8548, base_acc[1]],
            [0.9050, 0.9120, 0.9101, 0.9030, base_acc[2]],
        ]
    else: # MELSPEC
        detailed_metrics_per_fold = [
            [0.9050, 0.8900, 0.8980, 0.9100, base_acc[0]],
            [0.8800, 0.8950, 0.8900, 0.8750, base_acc[1]],
            [0.9150, 0.9200, 0.9100, 0.9100, base_acc[2]],
        ]


    actual_fold_outputs_for_summary = []

    for i, hive_id_key in enumerate(hives_in_order_sample):
        template_idx = i % len(detailed_metrics_per_fold)
        metrics_this_fold = detailed_metrics_per_fold[template_idx]
        
        acc_to_use = metrics_this_fold[4]
        loss_to_use = base_loss[template_idx] # Use base loss for simplicity

        # Add slight jitter if more hives than templates, to make them look different
        if i >= len(detailed_metrics_per_fold):
            acc_to_use = metrics_this_fold[4] + (random.random() - 0.5) * 0.01 # Smaller jitter
            loss_to_use = base_loss[template_idx] + (random.random() - 0.5) * 0.01
            acc_to_use = np.clip(acc_to_use, base_acc[template_idx]-0.02, base_acc[template_idx]+0.02)
            loss_to_use = np.clip(loss_to_use, base_loss[template_idx]-0.02, base_loss[template_idx]+0.02)


        test_size_this_fold = simulated_hive_info[hive_id_key]
        train_size_this_fold = simulated_total_samples - test_size_this_fold
        num_nq_this_fold = test_size_this_fold // 2
        num_q_this_fold = test_size_this_fold - num_nq_this_fold

        # Calculate F1 and averages for the report string
        nq_p, nq_r, q_p, q_r = metrics_this_fold[0], metrics_this_fold[1], metrics_this_fold[2], metrics_this_fold[3]
        nq_f1 = 2 * (nq_p * nq_r) / (nq_p + nq_r) if (nq_p + nq_r) > 0 else 0
        q_f1 = 2 * (q_p * q_r) / (q_p + q_r) if (q_p + q_r) > 0 else 0
        macro_p = (nq_p + q_p) / 2
        macro_r = (nq_r + q_r) / 2
        macro_f1 = (nq_f1 + q_f1) / 2
        w_avg_p = (nq_p * num_nq_this_fold + q_p * num_q_this_fold) / test_size_this_fold if test_size_this_fold > 0 else 0
        w_avg_r = (nq_r * num_nq_this_fold + q_r * num_q_this_fold) / test_size_this_fold if test_size_this_fold > 0 else 0
        w_avg_f1 = (nq_f1 * num_nq_this_fold + q_f1 * num_q_this_fold) / test_size_this_fold if test_size_this_fold > 0 else 0

        current_report = sample_fold_metrics_template[0]['report'].format(
            nq_p=nq_p, nq_r=nq_r, nq_f1=nq_f1, N_NQ=num_nq_this_fold,
            q_p=q_p, q_r=q_r, q_f1=q_f1, N_Q=num_q_this_fold,
            acc=acc_to_use, N_TOT=test_size_this_fold,
            macro_p=macro_p, macro_r=macro_r, macro_f1=macro_f1,
            w_avg_p=w_avg_p, w_avg_r=w_avg_r, w_avg_f1=w_avg_f1
        )
        
        print(f"\n--- Fold {i + 1}/{num_hives_sample}: Testing on Hive '{hive_id_key}' ---")
        print(f"Train size: {train_size_this_fold}, Test size: {test_size_this_fold}")
        print(f"Fold {i + 1} - Test on Hive '{hive_id_key}': Loss={loss_to_use:.4f}, Accuracy={acc_to_use:.4f}")
        print(f"--- Classification Report for Fold {i + 1} (Test Hive: {hive_id_key}) ---")
        print(current_report)
        
        model_fold_filename = f"best_model_{feature_name_display}_fold{i+1}_test_hive_{hive_id_key}.keras"
        model_fold_save_path = os.path.join(simulated_model_save_dir, model_fold_filename) 
        os.makedirs(os.path.dirname(model_fold_save_path), exist_ok=True) 
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
    print(f"=================== END OF SAMPLE FOR: {feature_name_display.upper()} ===================")