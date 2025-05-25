# utils_plot_v2.py
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import config_v2 as cfg
import config_v2 as cfg

def plot_waveform(audio_data, sample_rate, title="Waveform"): # <--- FUNCTION DEFINITION
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio_data, sr=sample_rate)
    plt.title(title); plt.xlabel('Time (s)'); plt.ylabel('Amplitude')
    plt.tight_layout(); plt.show()

def plot_waveform(audio_data, sample_rate, title="Waveform"):
    # ... (same as before) ...
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio_data, sr=sample_rate)
    plt.title(title); plt.xlabel('Time (s)'); plt.ylabel('Amplitude')
    plt.tight_layout(); plt.show()


def plot_feature_heatmap(feature_matrix, title="Feature Heatmap", sr=cfg.TARGET_SAMPLE_RATE, 
                        hop_length=cfg.HOP_LENGTH, feature_type="MFCC"):
    if feature_matrix.ndim == 3 and feature_matrix.shape[-1] == 1:
        feature_matrix = np.squeeze(feature_matrix, axis=-1)
    if feature_matrix.ndim != 2:
        print(f"Cannot plot: Matrix must be 2D. Got {feature_matrix.shape}"); return

    plt.figure(figsize=(10, 4))
    if feature_type.lower() == "melspec":
        librosa.display.specshow(feature_matrix, sr=sr, hop_length=hop_length, 
                                 x_axis='time', y_axis='mel', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
    else: # Assuming MFCC or similar
        librosa.display.specshow(feature_matrix, sr=sr, hop_length=hop_length, 
                                 x_axis='time', cmap='viridis')
        plt.colorbar() # No dB for MFCC generally
    plt.title(title); plt.tight_layout(); plt.show()


def plot_training_history_v2(history, save_dir, model_name="model"):
    # ... (same as your plot_utils.plot_training_history, but use save_dir and model_name from cfg if preferred) ...
    os.makedirs(save_dir, exist_ok=True)
    acc = history.history.get('accuracy')
    val_acc = history.history.get('val_accuracy')
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')
    epochs_range = range(1, len(acc) + 1 if acc else 1)
    plt.figure(figsize=(14, 5))
    if acc and val_acc:
        plt.subplot(1, 2, 1); plt.plot(epochs_range, acc, label='Training Acc'); plt.plot(epochs_range, val_acc, label='Validation Acc')
        plt.legend(loc='lower right'); plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    if loss and val_loss:
        plt.subplot(1, 2, 2); plt.plot(epochs_range, loss, label='Training Loss'); plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right'); plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.suptitle(f'History for {model_name}', fontsize=16); plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(save_dir, f'{model_name}_history.png')
    plt.savefig(plot_path); print(f"History plot saved to {plot_path}"); plt.show()

def evaluate_model_and_save_results(model, X_eval, y_eval, model_name_prefix, save_dir):
    # ... (same as your plot_utils.evaluate_and_plot_results, use class_names from cfg) ...
    os.makedirs(save_dir, exist_ok=True)
    loss, accuracy = model.evaluate(X_eval, y_eval, verbose=0)
    print(f"\n--- {model_name_prefix} Evaluation ---"); print(f"Loss: {loss:.4f}, Accuracy: {accuracy*100:.2f}%")
    y_pred_probs = model.predict(X_eval, verbose=0); y_pred_classes = np.argmax(y_pred_probs, axis=1)
    report = classification_report(y_eval, y_pred_classes, target_names=cfg.CLASS_NAMES)
    print("\nClassification Report:\n", report)
    report_path = os.path.join(save_dir, f'{model_name_prefix}_report.txt')
    with open(report_path, 'w') as f: f.write(report)
    print(f"Report saved to {report_path}")
    cm = confusion_matrix(y_eval, y_pred_classes)
    plt.figure(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cfg.CLASS_NAMES, yticklabels=cfg.CLASS_NAMES)
    plt.title(f'CM for {model_name_prefix}'); plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
    cm_path = os.path.join(save_dir, f'{model_name_prefix}_cm.png')
    plt.savefig(cm_path); print(f"CM plot saved to {cm_path}"); plt.show()
    return accuracy, classification_report(y_eval, y_pred_classes, target_names=cfg.CLASS_NAMES, output_dict=True)