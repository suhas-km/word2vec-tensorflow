import tensorflow as tf
import numpy as np

class Word2Vec(tf.keras.Model):
    """Skip-gram Word2Vec model implementation."""
    
    def __init__(self, vocab_size, embedding_dim):
        """Initialize Word2Vec model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
        """
        super(Word2Vec, self).__init__()
        self.target_embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            name="w2v_embedding",
        )
        self.context_embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim
        )
        
    def call(self, pair):
        """Forward pass of the model.
        
        Args:
            pair: Tuple of (target_word, context_word) tensors with shape (batch_size,)
            
        Returns:
            Dot product similarity scores with shape (batch_size, 1)
        """
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
    if tf.io.gfile.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_weights(model_path)
    else:
        print(f"No model found at {model_path}. Creating new model.")
    
    return model
