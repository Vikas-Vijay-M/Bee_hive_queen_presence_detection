# utils_predict_v2.py
import librosa
import numpy as np
import os
from tensorflow.keras.models import load_model as tf_load_model
import math
import config_v2 as cfg # For default constants

# Using _ naming for internal helpers
def _extract_mfcc_from_audio_data_pred(y_audio_segment, sr, n_mfcc, pad_to_frames):
    # ... (same as in your previous prediction_utils, using passed args) ...
    mfcc = librosa.feature.mfcc(y=y_audio_segment, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < pad_to_frames:
        pad_width = pad_to_frames - mfcc.shape[1]
        mfcc_padded = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc_padded = mfcc[:, :pad_to_frames]
    return mfcc_padded[..., np.newaxis]

def _extract_melspec_from_audio_data_pred(y_audio_segment, sr, n_mels, pad_to_frames, n_fft, hop_length):
    # ... (extract melspectrogram and pad/truncate) ...
    melspec = librosa.feature.melspectrogram(y=y_audio_segment, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    if melspec_db.shape[1] < pad_to_frames:
        pad_width = pad_to_frames - melspec_db.shape[1]
        melspec_padded = np.pad(melspec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        melspec_padded = melspec_db[:, :pad_to_frames]
    return melspec_padded[..., np.newaxis]


def _segment_audio_file_pred(file_path, chunk_duration_seconds, target_sr):
    # ... (same as _segment_audio_file in your previous prediction_utils) ...
    try:
        y_audio_full, sr_loaded = librosa.load(file_path, sr=target_sr)
    except Exception as e: raise IOError(f"Could not load {file_path}") from e
    total_duration_seconds = librosa.get_duration(y=y_audio_full, sr=sr_loaded)
    num_full_chunks = math.floor(total_duration_seconds / chunk_duration_seconds)
    audio_segments_data = []
    if num_full_chunks == 0:
        if total_duration_seconds > 0:
            samples_needed = int(target_sr * chunk_duration_seconds)
            if len(y_audio_full) < samples_needed: 
                y_padded_segment = librosa.util.fix_length(y_audio_full, size=samples_needed)
                audio_segments_data.append(y_padded_segment)
    else:
        samples_per_chunk = int(target_sr * chunk_duration_seconds)
        for i in range(num_full_chunks):
            start_sample = i * samples_per_chunk; end_sample = start_sample + samples_per_chunk
            audio_segments_data.append(y_audio_full[start_sample:end_sample])
    return audio_segments_data, target_sr


def classify_single_audio(file_path, model, feature_type_to_use):
    if not os.path.exists(file_path): raise FileNotFoundError(f"Not found: {file_path}")
    if model is None: raise ValueError("Model is None.")

    try:
        segments, sr = _segment_audio_file_pred(file_path, cfg.CHUNK_DURATION_SECONDS, cfg.TARGET_SAMPLE_RATE)
    except IOError as e: print(str(e)); return None, 0.0, 0, []
        
    if not segments: return None, 0.0, 0, []

    features_batch = []
    for seg_data in segments:
        if feature_type_to_use == cfg.MFCC_FEATURE_NAME:
            feat = _extract_mfcc_from_audio_data_pred(seg_data, sr, cfg.N_MFCC, cfg.MFCC_TARGET_FRAMES)
        elif feature_type_to_use == cfg.MELSPEC_FEATURE_NAME:
            feat = _extract_melspec_from_audio_data_pred(seg_data, sr, cfg.N_MELS, cfg.MELSPEC_TARGET_FRAMES, cfg.N_FFT, cfg.HOP_LENGTH)
        else:
            raise ValueError(f"Unknown feature_type_to_use: {feature_type_to_use}")
        features_batch.append(feat)
    
    if not features_batch: return None, 0.0, 0, []

    mfcc_np_batch = np.array(features_batch).astype(np.float32)
    
    # Dynamic shape check based on feature_type
    if feature_type_to_use == cfg.MFCC_FEATURE_NAME:
        expected_shape = cfg.MFCC_INPUT_SHAPE
    else: # MELSPEC
        expected_shape = cfg.MELSPEC_INPUT_SHAPE

    if mfcc_np_batch.ndim != 4 or mfcc_np_batch.shape[1:] != expected_shape:
        print(f"Shape error for {feature_type_to_use}. Got {mfcc_np_batch.shape}, expected (N, {expected_shape[0]}, {expected_shape[1]}, 1)")
        return None, 0.0, 0, []
        
    batch_preds = model.predict(mfcc_np_batch, verbose=0)
    avg_preds = np.mean(batch_preds, axis=0)
    pred_idx = np.argmax(avg_preds)
    conf = avg_preds[pred_idx]
    return pred_idx, conf, len(segments), batch_preds

def load_model_for_inference(model_name_prefix="mfcc"): # e.g. "mfcc" or "melspec"
    model_filename = f"best_model_{model_name_prefix}.keras"
    model_fpath = os.path.join(cfg.SAVED_MODEL_DIR, model_filename)
    if not os.path.exists(model_fpath):
        print(f"Error: Model '{model_fpath}' not found.")
        return None
    try:
        loaded_model = tf_load_model(model_fpath)
        print(f"Inference model '{model_fpath}' loaded.")
        return loaded_model
    except Exception as e:
        print(f"Error loading model '{model_fpath}': {e}")
        return None