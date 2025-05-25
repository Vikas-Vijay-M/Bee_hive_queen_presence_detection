# utils_model_v2.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, BatchNormalization, 
                                     GlobalAveragePooling2D, Dense, Dropout, Flatten)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, StratifiedKFold
import config_v2 as cfg # Your config file
# from utils_features_v2 import get_hive_id_from_filename # If used for grouping features directly


class EpochTestLog(Callback):
    def __init__(self, X_test, y_test, log_prefix="val"): # Changed log_prefix to match Keras val_
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.log_prefix = log_prefix

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        logs[f'{self.log_prefix}_loss'] = test_loss
        logs[f'{self.log_prefix}_accuracy'] = test_acc
        print(f" - {self.log_prefix}_loss: {test_loss:.4f} - {self.log_prefix}_accuracy: {test_acc:.4f}")

def build_advanced_cnn(input_shape, num_classes, learning_rate=cfg.LEARNING_RATE):
    model = Sequential([
        Input(shape=input_shape, name="input_layer"),
        
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Option 1: Flatten then Dense
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        # Option 2: GlobalAveragePooling (often better for image-like data)
        # GlobalAveragePooling2D(),
        # Dense(128, activation='relu'),
        # BatchNormalization(),
        # Dropout(0.5),

        Dense(num_classes, activation='softmax', name="output_layer")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_standard_model(X_train, y_train, X_test, y_test, feature_name):
    print(f"\n--- Training Standard Model with {feature_name.upper()} Features ---")
    
    if feature_name == cfg.MFCC_FEATURE_NAME:
        input_shape = cfg.MFCC_INPUT_SHAPE
    elif feature_name == cfg.MELSPEC_FEATURE_NAME:
        input_shape = cfg.MELSPEC_INPUT_SHAPE
    else:
        raise ValueError("Invalid feature_name for training.")

    model = build_advanced_cnn(input_shape, cfg.NUM_CLASSES)
    model_filename = f"best_model_{feature_name}.keras"
    model_save_path = os.path.join(cfg.SAVED_MODEL_DIR, model_filename)
    os.makedirs(cfg.SAVED_MODEL_DIR, exist_ok=True)

    checkpoint = ModelCheckpoint(filepath=model_save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    test_log_callback = EpochTestLog(X_test, y_test) # Use default val_ prefix

    print(f"Training with {len(X_train)} samples, validating on {len(X_test)} samples.")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=[checkpoint, early_stopping, reduce_lr, test_log_callback],
        verbose=1
    )
    
    # Load the best model for returning
    best_model = tf.keras.models.load_model(model_save_path)
    return best_model, history


# --- Leave-One-Hive-Out Cross-Validation ---
def train_and_evaluate_loho_cv(X_all, y_all, groups_all, feature_name):
    """
    Performs Leave-One-Hive-Out (LOCO) cross-validation.
    Args:
        X_all: All feature data (num_samples, height, width, 1)
        y_all: All labels
        groups_all: Array indicating group (hive) for each sample
        feature_name: Name of the feature type being used (for input shape)
    Returns:
        List of (history, test_accuracy, test_loss) for each fold.
    """
    if feature_name == cfg.MFCC_FEATURE_NAME:
        input_shape = cfg.MFCC_INPUT_SHAPE
    elif feature_name == cfg.MELSPEC_FEATURE_NAME:
        input_shape = cfg.MELSPEC_INPUT_SHAPE
    else:
        raise ValueError("Invalid feature_name for LOHO CV.")

    logo = LeaveOneGroupOut()
    fold_results = []
    
    unique_groups = np.unique(groups_all)
    print(f"\n--- Starting Leave-One-Hive-Out CV for {feature_name.upper()} ---")
    print(f"Total unique hives (groups): {len(unique_groups)}. Hives: {unique_groups}")

    for fold_idx, (train_indices, test_indices) in enumerate(logo.split(X_all, y_all, groups_all)):
        X_train_fold, X_test_fold = X_all[train_indices], X_all[test_indices]
        y_train_fold, y_test_fold = y_all[train_indices], y_all[test_indices]
        
        # Identify the hive being left out for testing
        # The group for all test_indices should be the same
        left_out_hive = groups_all[test_indices[0]] 
        
        print(f"\n--- Fold {fold_idx + 1}/{len(unique_groups)}: Testing on Hive '{left_out_hive}' ---")
        print(f"Train size: {len(X_train_fold)}, Test size: {len(X_test_fold)}")
        if len(np.unique(y_train_fold)) < cfg.NUM_CLASSES:
            print(f"Warning: Fold {fold_idx+1} training data has only {np.unique(y_train_fold)} classes. Skipping this fold.")
            fold_results.append({'history': None, 'test_acc': 0, 'test_loss': np.inf, 'left_out_hive': left_out_hive, 'status': 'skipped_insufficient_classes_in_train'})
            continue
        if len(X_test_fold) == 0:
            print(f"Warning: Fold {fold_idx+1} has no test data for hive {left_out_hive}. Skipping.")
            fold_results.append({'history': None, 'test_acc': 0, 'test_loss': np.inf, 'left_out_hive': left_out_hive, 'status': 'skipped_no_test_data'})
            continue


        model_fold = build_advanced_cnn(input_shape, cfg.NUM_CLASSES) # Fresh model for each fold
        
        model_fold_filename = f"best_model_{feature_name}_fold{fold_idx+1}_test_hive_{left_out_hive}.keras"
        model_fold_save_path = os.path.join(cfg.SAVED_MODEL_DIR, "loho_cv_models", model_fold_filename)
        os.makedirs(os.path.dirname(model_fold_save_path), exist_ok=True)

        checkpoint_fold = ModelCheckpoint(filepath=model_fold_save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=0) # Less verbose
        early_stopping_fold = EarlyStopping(monitor='val_loss', patience=8, verbose=0, restore_best_weights=True) # Shorter patience for CV
        reduce_lr_fold = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6, verbose=0)
        
        # Use a portion of the fold's training data as validation for callbacks
        # Simple split here for brevity; StratifiedShuffleSplit would be better
        val_split_idx = int(len(X_train_fold) * 0.9)
        X_train_for_fit, X_val_for_fit = X_train_fold[:val_split_idx], X_train_fold[val_split_idx:]
        y_train_for_fit, y_val_for_fit = y_train_fold[:val_split_idx], y_train_fold[val_split_idx:]
        
        if len(X_val_for_fit) == 0 or len(np.unique(y_val_for_fit)) < cfg.NUM_CLASSES :
            print(f"Warning: Fold {fold_idx+1} validation split has too few samples or classes. Using test set for validation during fit.")
            validation_data_for_fit = (X_test_fold, y_test_fold) # Fallback, not ideal for true LOHO stopping
        else:
             validation_data_for_fit = (X_val_for_fit, y_val_for_fit)


        history_fold = model_fold.fit(
            X_train_for_fit, y_train_for_fit,
            validation_data=validation_data_for_fit,
            epochs=cfg.EPOCHS, # Or a smaller number for CV for speed
            batch_size=cfg.BATCH_SIZE,
            callbacks=[checkpoint_fold, early_stopping_fold, reduce_lr_fold],
            verbose=0 # Less verbose for CV
        )
        
        # Load the best model for this fold
        best_model_fold = tf.keras.models.load_model(model_fold_save_path)
        loss_fold, acc_fold = best_model_fold.evaluate(X_test_fold, y_test_fold, verbose=0)
        print(f"Fold {fold_idx + 1} - Test on Hive '{left_out_hive}': Loss={loss_fold:.4f}, Accuracy={acc_fold:.4f}")
        
        fold_results.append({
            'history': history_fold, 
            'test_acc': acc_fold, 
            'test_loss': loss_fold,
            'left_out_hive': left_out_hive,
            'model_path': model_fold_save_path,
            'status': 'completed'
        })
        
    return fold_results