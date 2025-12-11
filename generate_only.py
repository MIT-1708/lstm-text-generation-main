import numpy as np
import tensorflow as tf
import string
import os
from tensorflow.keras.models import load_model

# Config
FILE_PATH = "shakespeare.txt"
SEQ_LENGTH = 40
TEST_SEEDS = ["love is", "the king said", "shall I"]

def preprocess_text(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        text = f.read()
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = ' '.join(text.split())
    return text

def get_mappings(text):
    chars = sorted(list(set(text)))
    char_indices = {c: i for i, c in enumerate(chars)}
    indices_char = {i: c for i, c in enumerate(chars)}
    return char_indices, indices_char

def generate_text(model, seed_text, length, char_indices, indices_char, seq_length):
    generated = seed_text
    translator = str.maketrans('', '', string.punctuation)
    clean_seed = seed_text.lower().translate(translator)
    current_sequence = clean_seed
    
    for _ in range(length):
        if len(current_sequence) < seq_length:
            pad_len = seq_length - len(current_sequence)
            input_seq_chars = (" " * pad_len) + current_sequence
        else:
            input_seq_chars = current_sequence[-seq_length:]
            
        x_pred = np.zeros((1, seq_length))
        for t, char in enumerate(input_seq_chars):
            idx = char_indices.get(char, 0) 
            x_pred[0, t] = idx
            
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = indices_char[next_index]
        
        generated += next_char
        current_sequence += next_char
        
    return generated

if __name__ == "__main__":
    text = preprocess_text(FILE_PATH)
    char_indices, indices_char = get_mappings(text)
    
    print("Loading model...")
    model = load_model('best_model.h5')
    
    print("Generating...")
    for seed in TEST_SEEDS:
        res = generate_text(model, seed, 100, char_indices, indices_char, SEQ_LENGTH)
        print(f"SEED: {seed}")
        print(f"RESULT: {res}")
        print("-" * 20)
