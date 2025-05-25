# app_v2.py

import streamlit as st
import os
import sys
from tempfile import NamedTemporaryFile
import numpy as np
import librosa # Needed for direct audio loading for plotting
import librosa.display # Needed for display functions
import matplotlib.pyplot as plt # Needed for plotting directly in Streamlit

# Import the audio recorder component
from streamlit_mic_recorder import mic_recorder 

# Ensure project modules can be found
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import config_v2 as cfg # Your configuration file
from utils_predict_v2 import (
    load_model_for_inference, # Renamed from load_keras_model_for_prediction
    classify_single_audio
)
# We might not need plot_utils_v2 here if we define plotting helpers directly or simplify

# --- Constants for UI and Interpretation ---
UI_CLASS_NAMES = {0: cfg.CLASS_NAMES[0], 1: cfg.CLASS_NAMES[1]}

st.set_page_config(page_title="Queen Bee Analyzer V2", layout="wide", initial_sidebar_state="auto")

st.title("👑 Advanced Queen Bee Analyzer 🐝")
st.markdown("Upload or record audio to classify queeee presence and visualize audio features.")
st.markdown("---")

# --- Load Models (Cached) ---
@st.cache_resource # Cache model loading
def get_models_for_app():
    models_dict = {}
    st.sidebar.write("Loading MFCC model...")
    models_dict[cfg.MFCC_FEATURE_NAME] = load_model_for_inference(model_name_prefix=cfg.MFCC_FEATURE_NAME)
    st.sidebar.write("Loading Mel-Spectrogram model...")
    models_dict[cfg.MELSPEC_FEATURE_NAME] = load_model_for_inference(model_name_prefix=cfg.MELSPEC_FEATURE_NAME)
    st.sidebar.write("Models loaded.")
    return models_dict

loaded_models = get_models_for_app()
available_features_for_prediction = [ftype for ftype, model in loaded_models.items() if model is not None]

# --- Sidebar for Prediction Options ---
st.sidebar.header("Prediction Options")
selected_feature_types_for_prediction = []
if cfg.MFCC_FEATURE_NAME in available_features_for_prediction:
    if st.sidebar.checkbox(f"Use MFCC Model for Prediction", value=True, key="use_mfcc_model_predict"):
        selected_feature_types_for_prediction.append(cfg.MFCC_FEATURE_NAME)
if cfg.MELSPEC_FEATURE_NAME in available_features_for_prediction:
    if st.sidebar.checkbox(f"Use Mel-Spectrogram Model for Prediction", value=True, key="use_melspec_model_predict"):
        selected_feature_types_for_prediction.append(cfg.MELSPEC_FEATURE_NAME)

# --- Main App Tabs ---
tab_classifier, tab_visualizer = st.tabs(["📊 Classifier", "🔎 Feature Visualizer"])

# ========================= CLASSIFIER TAB =========================
with tab_classifier:
    st.header("Classify Beehive Audio")
    input_method = st.radio("Choose input method:", ("📁 Upload Audio File", "🎤 Record Audio"), key="input_method_radio")

    def display_classification_results(predicted_idx, overall_confidence, num_segments, individual_probs, source_type="audio", feature_type_used="N/A"):
        st.markdown(f"##### Results using {feature_type_used.upper()} Model:")
        if predicted_idx is not None and num_segments > 0:
            final_prediction_text = UI_CLASS_NAMES.get(predicted_idx, "Unknown Prediction")
            if predicted_idx == 1: st.success(f"**{final_prediction_text}**"); 
            else: st.info(f"**{final_prediction_text}**")
            st.write(f"Overall Conf: **{overall_confidence * 100:.2f}%** ({num_segments} segs)")
            expander_title = f"Details for {feature_type_used.upper()} Segments ({source_type})"
            with st.expander(expander_title):
                for k, probs in enumerate(individual_probs):
                    s_idx=np.argmax(probs); s_conf=probs[s_idx]; s_label=UI_CLASS_NAMES.get(s_idx,"Unk")
                    st.caption(f"Seg {k+1}:'{s_label}',C:{s_conf*100:.1f}%,P:[NQ:{probs[0]*100:.0f}%,Q:{probs[1]*100:.0f}%]")
        elif num_segments == 0: st.warning(f"The {source_type} was too short for analysis with {feature_type_used.upper()}.")
        else: st.error(f"Prediction failed for {source_type} with {feature_type_used.upper()}.")

    temp_audio_file_for_classification = None # To store path of temp file for cleanup

    if input_method == "📁 Upload Audio File":
        uploaded_file_classifier = st.file_uploader("Upload a .wav file", type=["wav"], key="audio_uploader_classifier")
        if uploaded_file_classifier:
            st.audio(uploaded_file_classifier, format='audio/wav')
            with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_f:
                tmp_f.write(uploaded_file_classifier.getvalue())
                temp_audio_file_for_classification = tmp_f.name
    
    elif input_method == "🎤 Record Audio":
        st.caption(f"Try to record for at least {cfg.CHUNK_DURATION_SECONDS}s. Click 'Stop' when done.")
        audio_bytes_recorder = mic_recorder(start_prompt="🎤 Start Recording", stop_prompt="⏹️ Stop Recording", format="wav", key='recorder_classifier', use_container_width=True)
        if audio_bytes_recorder and audio_bytes_recorder['bytes']:
            st.audio(audio_bytes_recorder['bytes'], format='audio/wav')
            with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_f:
                tmp_f.write(audio_bytes_recorder['bytes'])
                temp_audio_file_for_classification = tmp_f.name

    if temp_audio_file_for_classification:
        if st.button("🚀 Analyze and Classify", key="analyze_classify_button", use_container_width=True, type="primary"):
            if not selected_feature_types_for_prediction:
                st.warning("No model selected for prediction in the sidebar. Please select at least one.")
            else:
                st.markdown("---")
                st.subheader(f"🔬 Prediction Results:")
                cols_classify = st.columns(len(selected_feature_types_for_prediction))

                for i, feature_type in enumerate(selected_feature_types_for_prediction):
                    model = loaded_models.get(feature_type)
                    if model:
                        with cols_classify[i]:
                            with st.spinner(f"Classifying with {feature_type.upper()}..."):
                                try:
                                    pred_idx, conf, num_seg, ind_probs = classify_single_audio(
                                        temp_audio_file_for_classification, model, feature_type
                                    )
                                    display_classification_results(pred_idx, conf, num_seg, ind_probs, 
                                                                   "this audio", feature_type)
                                except Exception as e:
                                    st.error(f"Error classifying with {feature_type.upper()}: {e}")
                    else:
                        with cols_classify[i]:
                            st.error(f"Model for {feature_type.upper()} not available.")
            
            # Cleanup temp file after analysis
            if os.path.exists(temp_audio_file_for_classification):
                try: os.remove(temp_audio_file_for_classification)
                except: pass # Ignore cleanup error if any

# ========================= FEATURE VISUALIZER TAB =========================
with tab_visualizer:
    st.header("Visualize Audio Features")
    st.write("Upload a short WAV audio clip (ideally around 2-5 seconds) to see its features.")
    
    uploaded_file_viz = st.file_uploader(
        "Upload a .wav file for visualization", 
        type=["wav"], 
        key="audio_uploader_visualizer"
    )

    if uploaded_file_viz is not None:
        st.subheader("🔊 Uploaded Audio for Visualization:")
        # For visualization, we might want to process the raw uploaded bytes directly
        # without necessarily saving to a NamedTemporaryFile if librosa can handle BytesIO
        audio_bytes_viz = uploaded_file_viz.getvalue()
        st.audio(audio_bytes_viz, format='audio/wav')

        # Load the audio for feature extraction
        # Using a BytesIO wrapper for librosa
        import io
        y_viz, sr_viz = librosa.load(io.BytesIO(audio_bytes_viz), sr=cfg.TARGET_SAMPLE_RATE)

        st.markdown("---")
        st.subheader("📈 Waveform Plot")
        fig_waveform, ax_waveform = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y_viz, sr=sr_viz, ax=ax_waveform)
        ax_waveform.set_title("Waveform")
        ax_waveform.set_xlabel("Time (s)")
        ax_waveform.set_ylabel("Amplitude")
        st.pyplot(fig_waveform)
        plt.close(fig_waveform) # Close the figure to free memory

        st.markdown("---")
        st.subheader(f"🎵 Mel-Frequency Cepstral Coefficients (MFCCs - {cfg.N_MFCC} coeffs)")
        # Extract MFCCs (no padding needed for visualization of the raw features)
        mfcc_viz = librosa.feature.mfcc(y=y_viz, sr=sr_viz, n_mfcc=cfg.N_MFCC, 
                                        n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH)
        fig_mfcc, ax_mfcc = plt.subplots(figsize=(10, 4))
        img_mfcc = librosa.display.specshow(mfcc_viz, sr=sr_viz, hop_length=cfg.HOP_LENGTH, 
                                            x_axis='time', ax=ax_mfcc)
        ax_mfcc.set_title(f"MFCC ({cfg.N_MFCC} Coefficients)")
        fig_mfcc.colorbar(img_mfcc, ax=ax_mfcc)
        st.pyplot(fig_mfcc)
        plt.close(fig_mfcc)

        st.markdown("---")
        st.subheader(f"🎶 Mel-Spectrogram ({cfg.N_MELS} Mel Bands)")
        # Extract Mel-Spectrogram
        melspec_viz = librosa.feature.melspectrogram(y=y_viz, sr=sr_viz, n_mels=cfg.N_MELS,
                                                     n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH)
        melspec_db_viz = librosa.power_to_db(melspec_viz, ref=np.max)
        fig_melspec, ax_melspec = plt.subplots(figsize=(10, 4))
        img_melspec = librosa.display.specshow(melspec_db_viz, sr=sr_viz, hop_length=cfg.HOP_LENGTH,
                                               x_axis='time', y_axis='mel', ax=ax_melspec)
        ax_melspec.set_title(f"Mel-Spectrogram ({cfg.N_MELS} Mel Bands)")
        fig_melspec.colorbar(img_melspec, ax=ax_melspec, format='%+2.0f dB')
        st.pyplot(fig_melspec)
        plt.close(fig_melspec)

        st.markdown("---")
        st.subheader("⚙️ Computational Aspects: MFCC vs. Mel-Spectrogram")
        st.markdown(f"""
        Both MFCCs and Mel-Spectrograms are popular for audio classification. They are derived from the Short-Time Fourier Transform (STFT).

        **1. Mel-Spectrogram:**
        *   **Process:** STFT -> Mel Filterbank -> Log Power (dB).
        *   **Representation:** A 2D representation of how the energy of different frequency bands (scaled to the Mel scale, which mimics human hearing) changes over time. It's like a spectrogram warped to be more sensitive to frequencies humans perceive better.
        *   **Dimensionality:** Typically `(n_mels, num_frames)`. For a {cfg.CHUNK_DURATION_SECONDS}s chunk at {cfg.TARGET_SAMPLE_RATE}Hz with {cfg.N_MELS} mels and hop length {cfg.HOP_LENGTH}, this would be roughly `({cfg.N_MELS}, {int((cfg.TARGET_SAMPLE_RATE * cfg.CHUNK_DURATION_SECONDS)/cfg.HOP_LENGTH) +1 })`.
        *   **Computational Cost:** Involves STFT computation and then Mel filterbank application. Generally moderate.
        *   **Information Content:** Retains a good amount of spectral detail, often beneficial for CNNs that can learn spatial patterns.

        **2. Mel-Frequency Cepstral Coefficients (MFCCs):**
        *   **Process:** STFT -> Mel Filterbank -> Log Power -> Discrete Cosine Transform (DCT).
        *   **Representation:** The DCT step decorrelates the filter bank coefficients and compacts the energy into the first few coefficients. Typically, only a small number of lower-order MFCCs (e.g., 13-40) are kept.
        *   **Dimensionality:** Typically `(n_mfcc, num_frames)`. For your setup: `({cfg.N_MFCC}, {int((cfg.TARGET_SAMPLE_RATE * cfg.CHUNK_DURATION_SECONDS)/cfg.HOP_LENGTH) +1 })`. This is significantly smaller in the frequency dimension than a Mel-spectrogram with many Mel bands.
        *   **Computational Cost:** Includes all steps of Mel-Spectrogram plus an additional DCT. The DCT itself is efficient (related to FFT).
        *   **Information Content:** More compressed representation. The DCT aims to capture the overall shape of the spectrum. Can be very effective but might discard some fine-grained details that a Mel-spectrogram retains. Often good for simpler models or when dimensionality reduction is desired.

        **Time Complexity (High-Level for a single frame/chunk analysis):**
        *   **STFT:** Dominated by FFTs. If window length is `N` and `K` frames, roughly `O(K * N log N)`.
        *   **Mel-Spectrogram:** STFT cost + cost of applying Mel filterbank (matrix multiplication, roughly `O(K * n_fft * n_mels)` if done naively, but often optimized).
        *   **MFCC:** Mel-Spectrogram cost + cost of DCT (efficient, similar to FFT, `O(K * n_mels log n_mels)`).

        **In practice for your setup:**
        *   Extracting a full Mel-Spectrogram with `{cfg.N_MELS}` bands will initially be slightly less work than extracting `{cfg.N_MFCC}` MFCCs because MFCCs require the extra DCT step.
        *   However, the resulting Mel-Spectrogram feature matrix (`{cfg.N_MELS} x {cfg.MELSPEC_TARGET_FRAMES}`) will be larger than the MFCC matrix (`{cfg.N_MFCC} x {cfg.MFCC_TARGET_FRAMES}`) if `{cfg.N_MELS}` > `{cfg.N_MFCC}`. This means a CNN processing Mel-spectrograms might have more input data per sample, potentially leading to longer training times per epoch or more parameters in the first convolutional layer, depending on the architecture.
        *   The key difference often comes down to what information the model can best leverage. Mel-spectrograms provide more raw spectral detail, while MFCCs provide a more compressed, decorrelated representation.
        """)

# --- Footer ---
st.markdown("---")
st.caption(f"SR={cfg.TARGET_SAMPLE_RATE}, Chunk={cfg.CHUNK_DURATION_SECONDS}s, MFCCs={cfg.N_MFCC}x{cfg.MFCC_TARGET_FRAMES}, MelSpec={cfg.N_MELS}x{cfg.MELSPEC_TARGET_FRAMES}")