# utils_chunk_v2.py
import os
import librosa
import soundfile as sf
import math
import config_v2 as cfg

def _chunk_audio_class_set(input_class_dir, output_class_dir, 
                           chunk_duration_seconds, target_sample_rate):
    os.makedirs(output_class_dir, exist_ok=True)
    if not os.path.exists(input_class_dir):
        print(f"Warning: Input directory {input_class_dir} not found. Skipping.")
        return 0
        
    num_chunks_created = 0
    for file_name in os.listdir(input_class_dir):
        if file_name.lower().endswith('.wav'):
            file_path = os.path.join(input_class_dir, file_name)
            try:
                y, sr = librosa.load(file_path, sr=target_sample_rate)
                total_duration = librosa.get_duration(y=y, sr=sr)
                num_possible_chunks = math.floor(total_duration / chunk_duration_seconds)
                
                for i in range(num_possible_chunks):
                    start_sample = int(i * chunk_duration_seconds * sr)
                    end_sample = int((i + 1) * chunk_duration_seconds * sr)
                    chunk = y[start_sample:end_sample]
                    
                    expected_length = int(chunk_duration_seconds * sr)
                    if len(chunk) < expected_length: # Pad if last chunk from floor is short
                        chunk = librosa.util.fix_length(chunk, size=expected_length)
                    elif len(chunk) > expected_length:
                         chunk = chunk[:expected_length]


                    chunk_filename = f"{os.path.splitext(file_name)[0]}_chunk{i}.wav"
                    chunk_filepath = os.path.join(output_class_dir, chunk_filename)
                    sf.write(chunk_filepath, chunk, sr)
                    num_chunks_created += 1
            except Exception as e:
                print(f"Error processing {file_path} for chunking: {e}")
    return num_chunks_created

def process_all_chunking():
    base_input_dir = cfg.DATA_SPLIT_DIR
    base_output_dir = cfg.DATA_CHUNKED_DIR
    chunk_duration_seconds = cfg.CHUNK_DURATION_SECONDS
    target_sample_rate = cfg.TARGET_SAMPLE_RATE
    class_names = cfg.CLASS_NAMES

    for set_type in ['train', 'test']:
        print(f"\nProcessing chunking for {set_type} set...")
        input_set_dir = os.path.join(base_input_dir, set_type)
        output_set_dir = os.path.join(base_output_dir, set_type)
        
        for class_label in class_names:
            input_class_dir = os.path.join(input_set_dir, class_label)
            output_class_dir = os.path.join(output_set_dir, class_label)
            print(f"  Chunking {class_label} files from {input_class_dir} to {output_class_dir}...")
            count = _chunk_audio_class_set(input_class_dir, output_class_dir, 
                                           chunk_duration_seconds, target_sample_rate)
            print(f"    Created {count} chunks for {class_label} in {set_type} set.")
    print("\n✅ Chunking completed for all sets and classes!")