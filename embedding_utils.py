import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def save_embeddings(model, vocab, filepath='word2vec_embeddings.npz'):
    """Save the trained word embeddings and vocabulary to file.
    
    Args:
        model: Trained Word2Vec model with target_embedding layer
        vocab: List of words in the vocabulary
        filepath: Path to save the embeddings
    """
    weights = model.get_layer('w2v_embedding').get_weights()[0]
    np.savez(filepath, weights=weights, vocab=vocab)
    print(f"Embeddings saved to {filepath}")
    return weights

def load_embeddings(filepath='word2vec_embeddings.npz'):
    """Load the trained word embeddings and vocabulary from file.
    
    Args:
        filepath: Path to load the embeddings from
        
    Returns:
        weights: Embedding weights matrix
        vocab: List of words in the vocabulary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Embedding file {filepath} not found")
    
    data = np.load(filepath, allow_pickle=True)
    weights = data['weights']
    vocab = data['vocab'].tolist()
    print(f"Loaded embeddings of shape {weights.shape} and vocabulary of size {len(vocab)}")
    return weights, vocab

def find_similar_words(word, weights, vocab, top_k=10):
    """Find most similar words based on cosine similarity.
    
    Args:
        word: Query word
        weights: Embedding weights matrix
        vocab: List of words in the vocabulary
        top_k: Number of similar words to return
        
    Returns:
        List of (word, similarity) tuples
    """
    if word not in vocab:
        print(f"'{word}' is not in vocabulary")
        return []
    
    word_idx = vocab.index(word)
    word_vec = weights[word_idx]
    word_vec = word_vec / np.linalg.norm(word_vec)  # Normalize
    
    similarities = []
    for i, v in enumerate(vocab):
        if i != word_idx:  # Skip the query word
            vec = weights[i]
            vec = vec / np.linalg.norm(vec)  # Normalize
            similarity = np.dot(word_vec, vec)
            similarities.append((v, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]

def visualize_embeddings(weights, vocab, words_to_plot=None, n_components=2):
    """Visualize word embeddings using t-SNE for dimensionality reduction.
    
    Args:
        weights: Embedding weights matrix
        vocab: List of words in vocabulary
        words_to_plot: List of specific words to highlight in the plot (optional)
        n_components: Dimensions for t-SNE (2 or 3)
    """
    # If no specific words provided, use most common words (excluding empty string and UNK)
    if words_to_plot is None:
        # Assume first words in vocab are most common (usually the case)
        words_to_plot = [w for w in vocab[2:52] if w]  # Skip '', '[UNK]' and get next 50
    
    # Filter to only words in vocabulary
    words_to_plot = [w for w in words_to_plot if w in vocab]
    
    if not words_to_plot:
        print("No valid words to plot")
        return
    
    # Get indices and embeddings for words to plot
    word_indices = [vocab.index(word) for word in words_to_plot]
    word_embeddings = weights[word_indices]
    
    # Apply t-SNE
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(30, max(5, len(words_to_plot)-1)))
    embeddings_2d = tsne.fit_transform(word_embeddings)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    if n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.figure(figsize=(12, 10)).add_subplot(111, projection='3d')
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], embeddings_2d[:, 2])
        
        for i, word in enumerate(words_to_plot):
            ax.text(embeddings_2d[i, 0], embeddings_2d[i, 1], embeddings_2d[i, 2], word)
            
    else:  # 2D plot
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
        
        for i, word in enumerate(words_to_plot):
            plt.annotate(word, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                        xytext=(5, 2), textcoords='offset points',
                        ha='right', va='bottom')
    
    plt.title(f"Word Embeddings Visualization using t-SNE ({n_components}D)")
    plt.tight_layout()
    plt.savefig(f'word_embeddings_{n_components}d.png')
    plt.show()

def word_analogy(word1, word2, word3, weights, vocab, top_k=5):
    """Perform word analogy task: word1 is to word2 as word3 is to ?
    
    Args:
        word1, word2, word3: Words for analogy
        weights: Embedding weights matrix
        vocab: List of words in vocabulary
        top_k: Number of top results to return
        
    Returns:
        List of (word, similarity) tuples
    """
    if not all(w in vocab for w in [word1, word2, word3]):
        missing = [w for w in [word1, word2, word3] if w not in vocab]
        print(f"Words not in vocabulary: {missing}")
        return []
    
    word1_idx, word2_idx, word3_idx = [vocab.index(w) for w in [word1, word2, word3]]
    
    # Calculate target vector: word2 - word1 + word3
    target_vec = weights[word2_idx] - weights[word1_idx] + weights[word3_idx]
    target_vec = target_vec / np.linalg.norm(target_vec)  # Normalize
    
    # Calculate similarity with all words
    similarities = []
    exclude_idxs = {word1_idx, word2_idx, word3_idx}
    
    for i, word in enumerate(vocab):
        if i not in exclude_idxs:
            vec = weights[i]
            vec = vec / np.linalg.norm(vec)  # Normalize
            similarity = np.dot(target_vec, vec)
            similarities.append((word, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]
