# app_v2.py

import streamlit as st
import os
import sys
from tempfile import NamedTemporaryFile
import numpy as np
from streamlit_mic_recorder import mic_recorder

# Ensure project root is in sys.path to find other modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import config_v2 as cfg # Your configuration file
from utils_predict_v2 import load_model_for_inference, classify_single_audio

# --- Constants for UI and Interpretation ---
UI_CLASS_NAMES = {0: cfg.CLASS_NAMES[0], 1: cfg.CLASS_NAMES[1]} # Using from config

st.set_page_config(page_title="Queen Bee Detector V2", layout="wide", initial_sidebar_state="auto")

st.title("👑 Advanced Queen Bee Detector 🐝")
st.markdown(
    "Upload a `.wav` audio file or record audio. The system will analyze it using models "
    "trained on different audio features (MFCC and Mel-Spectrogram) if available."
)
st.markdown("---")

# --- Load Models (Cached) ---
@st.cache_resource
def load_all_models():
    models = {}
    st.write("Loading MFCC model...")
    models[cfg.MFCC_FEATURE_NAME] = load_model_for_inference(model_name_prefix=cfg.MFCC_FEATURE_NAME)
    st.write("Loading Mel-Spectrogram model...")
    models[cfg.MELSPEC_FEATURE_NAME] = load_model_for_inference(model_name_prefix=cfg.MELSPEC_FEATURE_NAME)
    return models

loaded_models = load_all_models()

# Check which models loaded successfully
available_features_for_prediction = [ftype for ftype, model in loaded_models.items() if model is not None]

if not available_features_for_prediction:
    st.error(
        "No classification models could be loaded. Please ensure model files like "
        f"'saved_models/best_model_{cfg.MFCC_FEATURE_NAME}.keras' and "
        f"'saved_models/best_model_{cfg.MELSPEC_FEATURE_NAME}.keras' exist and are valid."
    )
else:
    st.sidebar.header("Prediction Options")
    # Allow user to select which feature types/models to use for prediction
    selected_feature_types = []
    if cfg.MFCC_FEATURE_NAME in available_features_for_prediction:
        if st.sidebar.checkbox(f"Use MFCC Model", value=True, key="use_mfcc_model"):
            selected_feature_types.append(cfg.MFCC_FEATURE_NAME)
    if cfg.MELSPEC_FEATURE_NAME in available_features_for_prediction:
        if st.sidebar.checkbox(f"Use Mel-Spectrogram Model", value=True, key="use_melspec_model"):
            selected_feature_types.append(cfg.MELSPEC_FEATURE_NAME)
    
    if not selected_feature_types:
        st.sidebar.warning("Please select at least one model type to use for prediction.")

    # --- Create tabs for different input methods ---
    tab1, tab2 = st.tabs(["📁 Upload Audio File", "🎤 Record Audio"])

    def process_and_display(audio_file_path_or_bytes, input_source_name):
        if not selected_feature_types:
            st.warning("No model selected for prediction in the sidebar.")
            return

        st.markdown("---")
        st.subheader(f"🔬 Prediction Results for {input_source_name}:")

        temp_file_to_delete = None
        audio_path_for_classification = None

        if isinstance(audio_file_path_or_bytes, str): # It's a file path
            audio_path_for_classification = audio_file_path_or_bytes
        else: # It's bytes (from recorder)
            with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_f:
                tmp_f.write(audio_file_path_or_bytes)
                audio_path_for_classification = tmp_f.name
                temp_file_to_delete = audio_path_for_classification
        
        cols = st.columns(len(selected_feature_types))

        for i, feature_type in enumerate(selected_feature_types):
            model_to_use = loaded_models.get(feature_type)
            if model_to_use is None:
                cols[i].error(f"Model for {feature_type.upper()} not loaded.")
                continue

            with cols[i]:
                st.markdown(f"#### Using {feature_type.upper()} Model")
                with st.spinner(f"Analyzing with {feature_type.upper()}..."):
                    try:
                        pred_idx, confidence, num_segments, ind_seg_probs = classify_single_audio(
                            audio_path_for_classification, 
                            model_to_use, 
                            feature_type # Pass the feature type to classify_single_audio
                        )

                        if pred_idx is not None and num_segments > 0:
                            pred_text = UI_CLASS_NAMES.get(pred_idx, "Unknown")
                            if pred_idx == 1: st.success(f"**{pred_text}**"); 
                            else: st.info(f"**{pred_text}**")
                            st.write(f"Conf: **{confidence * 100:.2f}%** ({num_segments} segs)")
                            
                            expander_title = f"Details for {feature_type.upper()} Segments"
                            with st.expander(expander_title):
                                for k, probs in enumerate(ind_seg_probs):
                                    s_idx=np.argmax(probs); s_conf=probs[s_idx]; s_label=UI_CLASS_NAMES.get(s_idx,"Unk")
                                    st.caption(f"Seg {k+1}:'{s_label}',C:{s_conf*100:.1f}%,P:[NQ:{probs[0]*100:.0f}%,Q:{probs[1]*100:.0f}%]")
                        elif num_segments == 0:
                            st.warning("Audio too short for analysis.")
                        else:
                            st.error("Prediction failed.")
                    except Exception as e:
                        st.error(f"Error ({feature_type}): {str(e)[:100]}...") # Show truncated error
        
        if temp_file_to_delete and os.path.exists(temp_file_to_delete):
            try: os.remove(temp_file_to_delete)
            except Exception as e: st.warning(f"Could not delete temp file: {e}")


    # --- Tab 1: File Uploader ---
    with tab1:
        st.subheader("Upload a WAV file:")
        uploaded_file = st.file_uploader(
            "Choose a .wav file", type=["wav"], key="audio_uploader_tab1_v2"
        )
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            if st.button("🚀 Analyze Uploaded Audio", key="analyze_uploaded_button_v2", use_container_width=True, type="primary"):
                # Save to a temporary file to get a path
                with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file_upload:
                    tmp_audio_file_upload.write(uploaded_file.getvalue())
                    temp_uploaded_path = tmp_audio_file_upload.name
                
                process_and_display(temp_uploaded_path, "Uploaded File")
                
                if os.path.exists(temp_uploaded_path): # Clean up
                    try: os.remove(temp_uploaded_path)
                    except Exception as e: st.warning(f"Could not delete temp upload file: {e}")
    
    # --- Tab 2: Microphone Recorder ---
    with tab2:
        st.subheader("Record audio from your microphone:")
        st.caption(f"Try to record for at least {cfg.CHUNK_DURATION_SECONDS} seconds. Click 'Stop Recording' when done.")
        
        # Using streamlit_mic_recorder
        audio_data_bytes = mic_recorder(
            start_prompt="🎤 Start Recording", 
            stop_prompt="🛑 Stop Recording", 
            format="wav", 
            key='recorder_v2',
            use_container_width=True
        ) 
        
        if audio_data_bytes and audio_data_bytes['bytes']:
            st.audio(audio_data_bytes['bytes'], format='audio/wav')
            if st.button("🚀 Analyze Recorded Audio", key="analyze_recorded_button_v2", use_container_width=True, type="primary"):
                process_and_display(audio_data_bytes['bytes'], "Recorded Audio")
        elif audio_data_bytes and not audio_data_bytes['bytes']:
            st.caption("No audio recorded or recording was too short.")

# --- Footer ---
st.markdown("---")
st.caption(f"Configuration: SR={cfg.TARGET_SAMPLE_RATE}, Chunk={cfg.CHUNK_DURATION_SECONDS}s, MFCC Frames={cfg.MFCC_TARGET_FRAMES}, MelSpec Frames={cfg.MELSPEC_TARGET_FRAMES}")
st.caption("Using TensorFlow, Librosa, streamlit-mic-recorder, and Streamlit.")