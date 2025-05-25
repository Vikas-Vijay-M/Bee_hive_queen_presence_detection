# config_v2.py
import os

# --- Data Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_NON_QUEEN_DIR = os.path.join(BASE_DIR, 'original_data', 'non_queen')
SOURCE_QUEEN_DIR = os.path.join(BASE_DIR, 'original_data', 'queen_present')

DATA_SPLIT_DIR = os.path.join(BASE_DIR, 'data_split')
DATA_CHUNKED_DIR = os.path.join(BASE_DIR, 'data_chunked')
SAVED_FEATURES_DIR = os.path.join(BASE_DIR, 'data_features')
SAVED_MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
EVALUATION_RESULTS_DIR = os.path.join(BASE_DIR, 'evaluation_results')
BEST_MODEL_NAME = 'best_model_mfcc.keras' # Default, will be feature-specific

# --- Audio Processing Parameters ---
TARGET_SAMPLE_RATE = 16000
CHUNK_DURATION_SECONDS = 2

# --- Feature Extraction Parameters ---
N_MFCC = 13
MFCC_TARGET_FRAMES = 63
MFCC_INPUT_SHAPE = (N_MFCC, MFCC_TARGET_FRAMES, 1)
MFCC_FEATURE_NAME = "mfcc"

N_MELS = 64
MELSPEC_TARGET_FRAMES = 63
MELSPEC_INPUT_SHAPE = (N_MELS, MELSPEC_TARGET_FRAMES, 1)
MELSPEC_FEATURE_NAME = "melspec"

N_FFT = 2048
HOP_LENGTH = 512

# Data Augmentation Parameters
NOISE_FACTOR = 0.005
PITCH_SHIFT_SEMITONES = [-1, 1]
TIME_STRETCH_RATES = [0.9, 1.1]
FREQ_MASK_PARAM = 15
TIME_MASK_PARAM = 20
NUM_MASKS = 1

# --- Model Training Parameters ---
NUM_CLASSES = 2
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001  # <--- ADD THIS LINE (or adjust value if needed)

# --- Labels and Class Names ---
LABELS_MAP = {'non_queen': 0, 'queen_present': 1}
CLASS_NAMES = ['non_queen', 'queen_present']



# config_v2.py
# ... (content from previous comprehensive response, ensure HIVE_IDS is accurate for your data) ...
HIVE_IDS = ['hive1', 'hive2', 'hive3', 'hive4', 'someotherhive'] # EXAMPLE - UPDATE THIS!