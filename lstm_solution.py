import numpy as np
import tensorflow as tf
import requests
import string
import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# 1. CONFIGURATION

URL = "https://www.gutenberg.org/files/100/100-0.txt"
FILE_PATH = "shakespeare.txt"
SEQ_LENGTH = 40
STEP_SIZE = 3  # Stride for sequence creation
BATCH_SIZE = 128
EPOCHS = 5  # Adjust based on time constraints
LSTM_UNITS = 256
EMBEDDING_DIM = 64
TEST_SEEDS = ["love is", "the king said", "shall I"]

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)


# 2. DATA LOADING & PREPROCESSING

def download_data(url, file_path):
    """Downloads the dataset if it doesn't exist."""
    if not os.path.exists(file_path):
        print(f"Downloading {url}...")
        response = requests.get(url)
        response.raise_for_status()
        # The text might have BOM or different encoding, utf-8-sig handles BOM
        with open(file_path, 'w', encoding='utf-8-sig') as f:
            f.write(response.text)
        print("Download complete.")
    else:
        print("File already exists.")

def preprocess_text(file_path):
    """Reads, cleans, and tokenizes text."""
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        text = f.read()
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    # Collapse whitespace
    text = ' '.join(text.split())
    
    # LIMIT DATASET SIZE FOR FASTER TRAINING ON CPU
    # The full text is ~5MB. We'll use the first 300k chars which is enough for a demo.
    limit = 300000
    if len(text) > limit:
        print(f"Truncating text from {len(text)} to {limit} characters for faster training.")
        text = text[:limit]
    
    print(f"Corpus length: {len(text)} characters")
    return text

def create_sequences(text, seq_length, step):
    """Creates input sequences and targets."""
    # Character-level tokenization
    chars = sorted(list(set(text)))
    char_indices = {c: i for i, c in enumerate(chars)}
    indices_char = {i: c for i, c in enumerate(chars)}
    vocab_size = len(chars)
    
    print(f"Unique characters: {vocab_size}")
    
    sentences = []
    next_chars = []
    
    for i in range(0, len(text) - seq_length, step):
        sentences.append(text[i : i + seq_length])
        next_chars.append(text[i + seq_length])
        
    print(f"Number of sequences: {len(sentences)}")
    
    # Vectorization
    # X: (num_sequences, seq_length) -> Indices
    # y: (num_sequences) -> Indices (Sparse target)
    
    X = np.zeros((len(sentences), seq_length), dtype=np.int32)
    y = np.zeros((len(sentences),), dtype=np.int32)
    
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t] = char_indices[char]
        y[i] = char_indices[next_chars[i]]
        
    return X, y, char_indices, indices_char, vocab_size


# 3. MODEL BUILDING

def build_model(vocab_size, seq_length, embedding_dim, lstm_units):
    """Builds the LSTM model."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_length),
        LSTM(lstm_units), # Default activation is tanh, recurrent_activation is sigmoid
        Dense(vocab_size, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# 4. TRAINING

def train_model(model, X, y):
    """Trains the model with EarlyStopping and Checkpoint."""
    
    # Split into train/validation manually or use validation_split
    # Using validation_split for simplicity
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
    ]
    
    history = model.fit(
        X, y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    return history

def plot_history(history):
    """Plots training and validation accuracy/loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")
    # plt.show() # Uncomment if running in a notebook environment


# 5. TEXT GENERATION

def generate_text(model, seed_text, length, char_indices, indices_char, seq_length):
    """Generates text given a seed."""
    
    generated = seed_text
    # Preprocess seed exactly like training data
    # Lowercase and remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    clean_seed = seed_text.lower().translate(translator)
    
    # Ensure seed is at least seq_length by padding if necessary (though requirements imply we provide valid seeds)
    # If seed is longer, take last seq_length
    # If seed is shorter, we can't easily predict without padding, but let's assume we just start filling.
    # However, the model expects fixed input length.
    
    current_sequence = clean_seed
    
    print(f"\n--- Generating for seed: '{seed_text}' ---")
    
    for _ in range(length):
        # Prepare input
        # Take last seq_length chars
        if len(current_sequence) < seq_length:
            pad_len = seq_length - len(current_sequence)
            input_seq_chars = (" " * pad_len) + current_sequence
        else:
            input_seq_chars = current_sequence[-seq_length:]
            
        # Vectorize
        x_pred = np.zeros((1, seq_length))
        for t, char in enumerate(input_seq_chars):
            # Handle unknown chars gracefully (though shouldn't happen with this dataset)
            idx = char_indices.get(char, 0) 
            x_pred[0, t] = idx
            
        # Predict
        preds = model.predict(x_pred, verbose=0)[0]
        
        # Greedy search (argmax) - deterministic
        # next_index = np.argmax(preds)
        
        # Sampling (Probabilistic) - more diverse
        # We can implement a simple temperature function or just use random choice
        next_index = np.random.choice(len(preds), p=preds)
        
        next_char = indices_char[next_index]
        
        generated += next_char
        current_sequence += next_char
        
    return generated



# MAIN EXECUTION

if __name__ == "__main__":
    # 1. Download
    download_data(URL, FILE_PATH)
    
    # 2. Preprocess
    text = preprocess_text(FILE_PATH)
    X, y, char_indices, indices_char, vocab_size = create_sequences(text, SEQ_LENGTH, STEP_SIZE)
    
    # 3. Build Model
    print("Building model...")
    model = build_model(vocab_size, SEQ_LENGTH, EMBEDDING_DIM, LSTM_UNITS)
    model.summary()
    
    # 4. Train
    print("Starting training...")
    history = train_model(model, X, y)
    plot_history(history)
    
    # 5. Generate
    print("\nGenerating text samples...")
    for seed in TEST_SEEDS:
        generated_text = generate_text(model, seed, length=100, 
                                       char_indices=char_indices, 
                                       indices_char=indices_char, 
                                       seq_length=SEQ_LENGTH)
        print(f"Result:\n{generated_text}\n")
        
    print("Done.")
