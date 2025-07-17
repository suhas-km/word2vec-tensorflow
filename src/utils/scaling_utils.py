import tensorflow as tf
import numpy as np
import os
import re
import string
import time
from tqdm.auto import tqdm

def preprocess_text(text):
    """Basic text preprocessing function for TensorFlow tensors.
    
    Args:
        text: String tensor to preprocess
        
    Returns:
        Preprocessed text tensor
    """
    # Convert to lowercase
    text = tf.strings.lower(text)
    
    # Replace punctuation with spaces (using regex)
    text = tf.strings.regex_replace(text, r'[!"#$%&\(\)\*\+,\-./:;<=>?@\[\\\]\^_`{\|}~]', ' ')
    
    # Remove extra whitespaces
    text = tf.strings.regex_replace(text, r'\s+', ' ')
    text = tf.strings.strip(text)
    
    return text

def create_text_vectorizer(corpus, vocab_size=4096, sequence_length=None):
    """Create and fit a text vectorizer layer.
    
    Args:
        corpus: List of text documents/sentences
        vocab_size: Size of vocabulary to create
        sequence_length: Max sequence length (None for dynamic)
        
    Returns:
        Fitted TextVectorization layer
    """
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_sequence_length=sequence_length,
        standardize=preprocess_text
    )
    
    # Create a text-only dataset (no labels)
    text_ds = tf.data.Dataset.from_tensor_slices(corpus).batch(1024)
    
    # Adapt the TextVectorization layer to the text
    vectorize_layer.adapt(text_ds)
    
    return vectorize_layer

def get_vocabulary_and_inverse(vectorize_layer):
    """Get vocabulary and inverse vocabulary mapping.
    
    Args:
        vectorize_layer: Fitted TextVectorization layer
        
    Returns:
        vocabulary: List of words in vocabulary
        inverse_vocabulary: Dict mapping word indices to words
    """
    vocabulary = vectorize_layer.get_vocabulary()
    inverse_vocabulary = dict(enumerate(vocabulary))
    
    return vocabulary, inverse_vocabulary

def load_large_corpus_from_directory(dir_path, file_extension='.txt', encoding='utf-8'):
    """Load multiple text files from a directory into a single corpus.
    
    Args:
        dir_path: Path to directory containing text files
        file_extension: File extension to look for
        encoding: Text encoding
        
    Returns:
        List of sentences from all files
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory {dir_path} not found")
    
    all_sentences = []
    files_processed = 0
    
    print(f"Loading files from {dir_path}...")
    
    # Walk through directory and process text files
    for root, _, files in os.walk(dir_path):
        for file in tqdm([f for f in files if f.endswith(file_extension)]):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                
                # Split into sentences (simple split by period)
                sentences = [s.strip() for s in text.split('.') if s.strip()]
                all_sentences.extend(sentences)
                files_processed += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    print(f"Processed {files_processed} files. Total sentences: {len(all_sentences)}")
    return all_sentences

def create_efficient_dataset(sequences, vocab_size, window_size=4, num_ns=4, 
                            batch_size=1024, buffer_size=10000, seed=42):
    """Create an efficient training dataset with tf.data API for large datasets.
    
    Args:
        sequences: List of integer sequences (already vectorized)
        vocab_size: Size of vocabulary
        window_size: Context window size (each side)
        num_ns: Number of negative samples per target-context pair
        batch_size: Batch size for training
        buffer_size: Buffer size for shuffling
        seed: Random seed
        
    Returns:
        TensorFlow dataset ready for training
    """
    # Generate skip-gram pairs with negative sampling
    targets, contexts, labels = generate_training_data(
        sequences=sequences, 
        window_size=window_size, 
        num_ns=num_ns, 
        vocab_size=vocab_size, 
        seed=seed
    )
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(buffer_size, seed=seed)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def generate_training_data(sequences, window_size, num_ns, vocab_size, seed=42):
    """Generate training data for Word2Vec model.
    
    Args:
        sequences: List of integer sequences (already vectorized)
        window_size: Context window size (each side)
        num_ns: Number of negative samples per target-context pair
        vocab_size: Size of vocabulary
        seed: Random seed
        
    Returns:
        targets: Array of target word indices
        contexts: Array of context word indices
        labels: Array of labels (1 for positive, 0 for negative)
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Initialize lists to store results
    targets, contexts, labels = [], [], []
    
    # Maximum number of words to process (set high by default)
    max_words = sum(len(sequence) for sequence in sequences)
    
    # Process each sequence (sentence/document)
    sample_table = get_sampling_table(vocab_size)
    progress_bar = tqdm(total=max_words, desc="Generating training examples")
    
    for sequence in sequences:
        # Skip sequences that are too short
        if len(sequence) < 2 * window_size + 1:
            continue
        
        # Process each word in the sequence
        for i in range(len(sequence)):
            progress_bar.update(1)
            
            # Get target word
            target = sequence[i]
            
            # Skip target words that are padding or unknown
            if target < 2:  # 0: padding, 1: unknown
                continue
            
            # Generate positive context words in window
            context_window = list(range(max(0, i - window_size), i)) + \
                             list(range(i + 1, min(len(sequence), i + window_size + 1)))
            
            for context_idx in context_window:
                context = sequence[context_idx]
                
                # Skip context words that are padding or unknown
                if context < 2:  # 0: padding, 1: unknown
                    continue
                
                # Add positive pair
                targets.append(target)
                contexts.append(context)
                labels.append(1)
                
                # Sample negative contexts
                negative_contexts = sample_negative_contexts(
                    target, context, num_ns, vocab_size, sample_table)
                
                # Add negative pairs
                targets.extend([target] * num_ns)
                contexts.extend(negative_contexts)
                labels.extend([0] * num_ns)
    
    progress_bar.close()
    
    # Convert lists to numpy arrays
    targets = np.array(targets, dtype=np.int64)
    contexts = np.array(contexts, dtype=np.int64)
    labels = np.array(labels, dtype=np.int64)
    
    print(f"Generated {len(targets)} total training examples")
    return targets, contexts, labels

def get_sampling_table(vocab_size, sampling_factor=0.75):
    """Create a table for unigram distribution sampling.
    
    Args:
        vocab_size: Size of vocabulary
        sampling_factor: Sampling factor for unigram distribution
    
    Returns:
        Sampling probability table
    """
    # Create a power distribution based on word frequency
    # Words with lower indices are assumed to be more frequent
    # This is a simple approximation
    counts = np.array(range(2, vocab_size))  # Skip 0 (padding) and 1 (UNK)
    p = counts ** (-sampling_factor)
    p = p / np.sum(p)
    
    # Prepend zeros for padding and UNK tokens
    return np.concatenate([np.zeros(2), p])

def sample_negative_contexts(target, positive_context, num_ns, vocab_size, sample_table):
    """Sample negative contexts for a target word.
    
    Args:
        target: Target word index
        positive_context: Positive context word index to avoid
        num_ns: Number of negative samples to generate
        vocab_size: Size of vocabulary
        sample_table: Sampling probability table
        
    Returns:
        List of negative context word indices
    """
    # Convert sample table to probabilities
    p = sample_table / np.sum(sample_table)
    
    # Sample from the distribution
    negative_samples = []
    while len(negative_samples) < num_ns:
        sample = np.random.choice(vocab_size, p=p)
        
        # Skip if it's the target or positive context
        if sample != target and sample != positive_context and sample >= 2:
            negative_samples.append(sample)
    
    return negative_samples

def train_with_checkpointing(model, dataset, epochs=5, checkpoint_dir='./checkpoints',
                           checkpoint_freq=1, tensorboard_dir='./logs'):
    """Train model with checkpointing and TensorBoard logging.
    
    Args:
        model: Compiled Word2Vec model
        dataset: TensorFlow dataset for training
        epochs: Number of training epochs
        checkpoint_dir: Directory to save checkpoints
        checkpoint_freq: Frequency of checkpoints (in epochs)
        tensorboard_dir: Directory for TensorBoard logs
        
    Returns:
        Training history
    """
    # Create directories if they don't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Define callbacks
    callbacks = [
        # TensorBoard callback
        tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=1),
        
        # Model checkpoint callback
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'word2vec_{epoch:02d}.weights.h5'),
            save_weights_only=True,
            save_freq=checkpoint_freq * len(dataset),
            verbose=1
        ),
        
        # Early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train the model
    start_time = time.time()
    history = model.fit(
        dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    elapsed = time.time() - start_time
    
    print(f"Training completed in {elapsed:.2f} seconds")
    return history
