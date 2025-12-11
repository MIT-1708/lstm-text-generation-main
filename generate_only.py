import argparse
import numpy as np
import tensorflow as tf
import string
import os
from tensorflow.keras.models import load_model

# Config
FILE_PATH = os.path.join(os.path.dirname(__file__), "shakespeare.txt")
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

def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / float(temperature)
    exp_preds = np.exp(preds)
    probs = exp_preds / np.sum(exp_preds)
    return probs


def generate_text(model, seed_text, length, char_indices, indices_char, seq_length, temperature=1.0):
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

        x_pred = np.zeros((1, seq_length), dtype=np.int32)
        for t, char in enumerate(input_seq_chars):
            idx = char_indices.get(char, 0)
            x_pred[0, t] = idx

        preds = model.predict(x_pred, verbose=0)[0]

        # Sample using temperature to avoid deterministic repetition
        probs = sample_with_temperature(preds, temperature=temperature)
        next_index = np.random.choice(len(probs), p=probs)
        next_char = indices_char[next_index]

        generated += next_char
        current_sequence += next_char

    return generated

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from trained LSTM model")
    parser.add_argument("--length", type=int, default=100, help="Number of characters to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0.2..1.2+)")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed phrases to use")
    args = parser.parse_args()

    text = preprocess_text(FILE_PATH)
    char_indices, indices_char = get_mappings(text)

    print("Loading model...")
    model_path = os.path.join(os.path.dirname(__file__), 'best_model.h5')
    model = load_model(model_path)

    seeds = TEST_SEEDS
    if args.seeds:
        seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]

    print("Generating...")
    for seed in seeds:
        res = generate_text(model, seed, args.length, char_indices, indices_char, SEQ_LENGTH, temperature=args.temperature)
        print(f"SEED: {seed}")
        print(f"RESULT: {res}")
        print("-" * 20)
