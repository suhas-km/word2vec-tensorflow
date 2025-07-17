import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from embedding_utils import (
    save_embeddings, load_embeddings, find_similar_words,
    visualize_embeddings, word_analogy
)
from scaling_utils import (
    load_large_corpus_from_directory, create_text_vectorizer,
    get_vocabulary_and_inverse, create_efficient_dataset,
    train_with_checkpointing
)

# Set random seed for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
AUTOTUNE = tf.data.AUTOTUNE

class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.target_embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=1,
            name="w2v_embedding",
        )
        self.context_embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=1
        )
        
    def call(self, pair):
        target, context = pair
        
        # Target shape: (batch,)
        # Context shape: (batch,)
        # Output shape after embedding: (batch, embedding_dim)
        
        target_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        
        # No need to squeeze as the shape is already (batch, embedding_dim)
        
        # Dot product between target and context embeddings
        # Output shape: (batch, 1)
        dots = tf.expand_dims(tf.reduce_sum(target_emb * context_emb, axis=1), axis=1)
        
        return dots


def create_model(vocab_size, embedding_dim):
    """Create and compile the Word2Vec model."""
    word2vec = Word2Vec(vocab_size, embedding_dim)
    
    # Custom loss function for Word2Vec
    def custom_loss(y_true, y_pred):
        return tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(y_true, tf.float32),
            logits=y_pred
        )
    
    # Compile the model with Adam optimizer
    word2vec.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=custom_loss
    )
    
    return word2vec


def load_or_create_model(model_path, vocab_size, embedding_dim):
    """Load an existing model or create a new one."""
    model = create_model(vocab_size, embedding_dim)
    
    # Build the model with some dummy data to initialize weights
    dummy_target = tf.constant([[1]])
    dummy_context = tf.constant([[2]])
    model((dummy_target, dummy_context))
    
    # Check if model weights exist
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_weights(model_path)
    else:
        print(f"No model found at {model_path}. Creating new model.")
    
    return model


def evaluate_embeddings(model, vocabulary, inverse_vocab, embedding_path=None):
    """Evaluate and visualize the trained word embeddings."""
    # Save embeddings to file
    weights = save_embeddings(model, vocabulary, embedding_path)
    
    # Find similar words examples
    common_words = ["king", "queen", "man", "woman", "love", "hate"]
    for word in common_words:
        if word in vocabulary:
            print(f"\nWords most similar to '{word}':")
            similar_words = find_similar_words(word, weights, vocabulary, top_k=5)
            for similar_word, similarity in similar_words:
                print(f"  {similar_word}: {similarity:.4f}")
        else:
            print(f"'{word}' not in vocabulary")
    
    # Word analogies
    analogies = [
        ("king", "man", "queen"),  # king - man + queen = woman
        ("man", "king", "woman"),  # man - king + woman = queen
    ]
    
    print("\nWord analogies:")
    for word1, word2, word3 in analogies:
        if all(word in vocabulary for word in [word1, word2, word3]):
            print(f"\n{word1} is to {word2} as {word3} is to:")
            results = word_analogy(word1, word2, word3, weights, vocabulary, top_k=5)
            for result_word, similarity in results:
                print(f"  {result_word}: {similarity:.4f}")
    
    # Visualize embeddings
    print("\nVisualizing word embeddings...")
    common_words = [w for w in common_words if w in vocabulary]
    
    # Add some additional words if available
    extra_words = ["good", "bad", "happy", "sad", "day", "night", "sun", "moon"]
    visualization_words = common_words + [w for w in extra_words if w in vocabulary]
    
    # Ensure we have enough words to visualize
    if len(visualization_words) > 10:
        visualize_embeddings(weights, vocabulary, visualization_words)
    else:
        # Use most common words if not enough specific words in vocabulary
        visualize_embeddings(weights, vocabulary)


def train_on_shakespeare(model_path='word2vec_shakespeare.weights.h5', embedding_dim=128, epochs=5):
    """Train Word2Vec model on Shakespeare text."""
    # Load Shakespeare text
    file_path = tf.keras.utils.get_file(
        'shakespeare.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
    )
    
    with open(file_path, 'r') as f:
        shakespeare_text = f.read()
    
    # Preprocess text into sentences
    sentences = shakespeare_text.lower().split('\n')
    sentences = [s.strip() for s in sentences if s.strip()]
    print(f"Loaded {len(sentences)} sentences from Shakespeare text")
    
    # Create text vectorizer
    vocab_size = 4096
    vectorize_layer = create_text_vectorizer(sentences, vocab_size)
    
    # Get vocabulary
    vocabulary, inverse_vocab = get_vocabulary_and_inverse(vectorize_layer)
    
    # Vectorize sentences
    sequences = [vectorize_layer(s) for s in sentences]
    
    # Create training dataset
    dataset = create_efficient_dataset(
        sequences, 
        vocab_size=vocab_size, 
        window_size=4, 
        num_ns=4, 
        batch_size=1024
    )
    
    # Create or load model
    model = load_or_create_model(model_path, vocab_size, embedding_dim)
    
    # Train model
    train_with_checkpointing(
        model, 
        dataset, 
        epochs=epochs, 
        checkpoint_dir='./shakespeare_checkpoints'
    )
    
    # Save final model
    model.save_weights(model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate embeddings
    evaluate_embeddings(model, vocabulary, inverse_vocab, 'shakespeare_embeddings.npz')
    
    return model, vocabulary


def train_on_large_corpus(corpus_dir, model_path='word2vec_large.weights.h5', 
                        embedding_dim=300, vocab_size=10000, epochs=5):
    """Train Word2Vec model on a large corpus."""
    # Check if directory exists
    if not os.path.exists(corpus_dir):
        print(f"Error: Directory '{corpus_dir}' not found")
        print("Please provide a valid directory path containing text files.")
        return None, None
        
    # Load text from directory
    try:
        sentences = load_large_corpus_from_directory(corpus_dir)
        if len(sentences) == 0:
            print(f"No text files found in '{corpus_dir}' or files were empty")
            return None, None
            
        print(f"Loaded {len(sentences)} sentences from {corpus_dir}")
    except Exception as e:
        print(f"Error loading corpus: {e}")
        return None, None
    
    # Create text vectorizer
    vectorize_layer = create_text_vectorizer(sentences, vocab_size)
    
    # Get vocabulary
    vocabulary, inverse_vocab = get_vocabulary_and_inverse(vectorize_layer)
    
    # Vectorize sentences
    sequences = [vectorize_layer(s) for s in sentences]
    
    # Create training dataset
    dataset = create_efficient_dataset(
        sequences, 
        vocab_size=vocab_size, 
        window_size=5, 
        num_ns=5, 
        batch_size=2048,
        buffer_size=50000
    )
    
    # Create or load model
    model = load_or_create_model(model_path, vocab_size, embedding_dim)
    
    # Train model
    train_with_checkpointing(
        model, 
        dataset, 
        epochs=epochs, 
        checkpoint_dir='./large_corpus_checkpoints'
    )
    
    # Save final model
    model.save_weights(model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate embeddings
    evaluate_embeddings(model, vocabulary, inverse_vocab, 'large_corpus_embeddings.npz')
    
    return model, vocabulary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and evaluate Word2Vec models")
    parser.add_argument("--mode", type=str, choices=["shakespeare", "large"], 
                      default="shakespeare", help="Training mode")
    parser.add_argument("--corpus-dir", type=str, default=None, 
                      help="Directory containing text corpus for large mode")
    parser.add_argument("--embedding-dim", type=int, default=128, 
                      help="Embedding dimension")
    parser.add_argument("--vocab-size", type=int, default=4096, 
                      help="Vocabulary size")
    parser.add_argument("--epochs", type=int, default=5, 
                      help="Number of training epochs")
    
    args = parser.parse_args()
    
    if args.mode == "shakespeare":
        print("Training Word2Vec on Shakespeare corpus...")
        model, vocab = train_on_shakespeare(
            embedding_dim=args.embedding_dim,
            epochs=args.epochs
        )
    else:
        if not args.corpus_dir:
            print("Error: corpus-dir must be provided for large mode")
            exit(1)
        
        print(f"Training Word2Vec on large corpus from {args.corpus_dir}...")
        model, vocab = train_on_large_corpus(
            args.corpus_dir,
            embedding_dim=args.embedding_dim,
            vocab_size=args.vocab_size,
            epochs=args.epochs
        )
