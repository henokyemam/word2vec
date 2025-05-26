"""
Simple Word2Vec Model Implementation - Educational Implementation
Step 6: Neural Network Architecture (Skip-gram and CBOW)

This module implements the Word2Vec model architectures with:
- Skip-gram: Predicts context words from center word
- CBOW: Predicts center word from context words
- Negative sampling for efficient training
- Clear, educational code structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import math


class Word2VecEmbeddings(nn.Module):
    """
    Base embedding layer for Word2Vec models.
    Contains the core word embedding matrix that we want to learn.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initialize word embeddings.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Input embeddings (the main embeddings we care about)
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Output embeddings (used for prediction, discarded after training)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize embeddings with small random values
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embeddings with small random values."""
        # Xavier uniform initialization scaled down
        bound = 0.5 / self.embedding_dim
        
        # Initialize input embeddings (these are the final word vectors)
        nn.init.uniform_(self.in_embeddings.weight.data, -bound, bound)
        
        # Initialize output embeddings (auxiliary vectors for training)
        nn.init.uniform_(self.out_embeddings.weight.data, -bound, bound)
    
    def get_word_embeddings(self) -> torch.Tensor:
        """
        Get the learned word embeddings (input embeddings).
        
        Returns:
            Tensor of shape (vocab_size, embedding_dim)
        """
        return self.in_embeddings.weight.data
    
    def get_word_embedding(self, word_id: int) -> torch.Tensor:
        """
        Get embedding for a specific word.
        
        Args:
            word_id: Word index
            
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        return self.in_embeddings.weight.data[word_id]


class SkipGramModel(nn.Module):
    """
    Skip-gram model: Predicts context words from center word.
    
    Architecture:
    Input: center word index → Embedding → Linear → Output probabilities
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initialize Skip-gram model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.model_type = 'skipgram'
        
        # Shared embeddings
        self.embeddings = Word2VecEmbeddings(vocab_size, embedding_dim)
        
        print(f"\n{'='*60}")
        print(f"SKIP-GRAM MODEL INITIALIZED")
        print(f"{'='*60}")
        print(f"Vocabulary size: {vocab_size:,}")
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Total parameters: {self.count_parameters():,}")
        print("-" * 60)
    
    def forward(self, center_words: torch.Tensor, 
                context_words: torch.Tensor,
                negative_words: Optional[torch.Tensor] = None,
                negative_centers: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Skip-gram model.
        
        Args:
            center_words: Center word indices [batch_size]
            context_words: Context word indices [batch_size]
            negative_words: Negative word indices [num_negatives] (optional)
            negative_centers: Center words for negatives [num_negatives] (optional)
            
        Returns:
            Dictionary with loss components and predictions
        """
        batch_size = center_words.size(0)
        
        # Get embeddings for center words
        center_embeds = self.embeddings.in_embeddings(center_words)  # [batch_size, embed_dim]
        
        # Positive samples: predict context words from center words
        context_out_embeds = self.embeddings.out_embeddings(context_words)  # [batch_size, embed_dim]
        positive_scores = torch.sum(center_embeds * context_out_embeds, dim=1)  # [batch_size]
        
        # Apply sigmoid to get probabilities
        positive_loss = F.logsigmoid(positive_scores).mean()
        
        results = {
            'positive_loss': positive_loss,
            'positive_scores': positive_scores,
            'center_embeddings': center_embeds
        }
        
        # Negative sampling if provided
        if negative_words is not None and negative_centers is not None:
            # Get embeddings for negative samples
            neg_center_embeds = self.embeddings.in_embeddings(negative_centers)  # [num_negatives, embed_dim]
            neg_out_embeds = self.embeddings.out_embeddings(negative_words)      # [num_negatives, embed_dim]
            
            # Negative scores (we want these to be low)
            negative_scores = torch.sum(neg_center_embeds * neg_out_embeds, dim=1)  # [num_negatives]
            negative_loss = F.logsigmoid(-negative_scores).mean()
            
            # Total loss
            total_loss = -(positive_loss + negative_loss)
            
            results.update({
                'negative_loss': negative_loss,
                'negative_scores': negative_scores,
                'total_loss': total_loss
            })
        else:
            # Without negative sampling, use full softmax (less efficient)
            all_out_embeds = self.embeddings.out_embeddings.weight  # [vocab_size, embed_dim]
            logits = torch.matmul(center_embeds, all_out_embeds.t())  # [batch_size, vocab_size]
            
            softmax_loss = F.cross_entropy(logits, context_words)
            results.update({
                'softmax_loss': softmax_loss,
                'logits': logits,
                'total_loss': softmax_loss
            })
        
        return results
    
    def predict_context(self, center_word: torch.Tensor, top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict most likely context words for a given center word.
        
        Args:
            center_word: Center word index [1]
            top_k: Number of top predictions to return
            
        Returns:
            Tuple of (top_k_indices, top_k_scores)
        """
        self.eval()
        with torch.no_grad():
            center_embed = self.embeddings.in_embeddings(center_word)  # [1, embed_dim]
            all_out_embeds = self.embeddings.out_embeddings.weight     # [vocab_size, embed_dim]
            
            # Compute similarity scores
            scores = torch.matmul(center_embed, all_out_embeds.t()).squeeze()  # [vocab_size]
            
            # Get top-k predictions
            top_scores, top_indices = torch.topk(scores, top_k)
            
        return top_indices, top_scores
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CBOWModel(nn.Module):
    """
    CBOW (Continuous Bag of Words) model: Predicts center word from context words.
    
    Architecture:
    Input: context word indices → Average embeddings → Linear → Output probabilities
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initialize CBOW model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.model_type = 'cbow'
        
        # Shared embeddings
        self.embeddings = Word2VecEmbeddings(vocab_size, embedding_dim)
        
        print(f"\n{'='*60}")
        print(f"CBOW MODEL INITIALIZED")
        print(f"{'='*60}")
        print(f"Vocabulary size: {vocab_size:,}")
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Total parameters: {self.count_parameters():,}")
        print("-" * 60)
    
    def forward(self, context_words: torch.Tensor,
                context_mask: torch.Tensor,
                center_words: torch.Tensor,
                negative_words: Optional[torch.Tensor] = None,
                negative_contexts: Optional[torch.Tensor] = None,
                negative_masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for CBOW model.
        
        Args:
            context_words: Context word indices [batch_size, max_context_len]
            context_mask: Mask for valid context words [batch_size, max_context_len]
            center_words: Center word indices [batch_size]
            negative_words: Negative word indices [num_negatives] (optional)
            negative_contexts: Context words for negatives [num_negatives, max_context_len] (optional)
            negative_masks: Masks for negative contexts [num_negatives, max_context_len] (optional)
            
        Returns:
            Dictionary with loss components and predictions
        """
        batch_size = context_words.size(0)
        
        # Get context embeddings
        context_embeds = self.embeddings.in_embeddings(context_words)  # [batch_size, max_context_len, embed_dim]
        
        # Apply mask and average context embeddings
        masked_embeds = context_embeds * context_mask.unsqueeze(-1)  # [batch_size, max_context_len, embed_dim]
        context_lengths = context_mask.sum(dim=1, keepdim=True)      # [batch_size, 1]
        context_lengths = torch.clamp(context_lengths, min=1)        # Avoid division by zero
        
        avg_context_embeds = masked_embeds.sum(dim=1) / context_lengths  # [batch_size, embed_dim]
        
        # Positive samples: predict center word from context
        center_out_embeds = self.embeddings.out_embeddings(center_words)  # [batch_size, embed_dim]
        positive_scores = torch.sum(avg_context_embeds * center_out_embeds, dim=1)  # [batch_size]
        
        # Apply sigmoid to get probabilities
        positive_loss = F.logsigmoid(positive_scores).mean()
        
        results = {
            'positive_loss': positive_loss,
            'positive_scores': positive_scores,
            'context_embeddings': avg_context_embeds
        }
        
        # Negative sampling if provided
        if negative_words is not None and negative_contexts is not None and negative_masks is not None:
            # Ensure we have the same number of negative words and contexts
            assert negative_words.size(0) == negative_contexts.size(0), f"Negative words ({negative_words.size(0)}) and contexts ({negative_contexts.size(0)}) must have same batch size"
            assert negative_words.size(0) == negative_masks.size(0), f"Negative words ({negative_words.size(0)}) and masks ({negative_masks.size(0)}) must have same batch size"
            
            # Process negative contexts
            neg_context_embeds = self.embeddings.in_embeddings(negative_contexts)  # [num_negatives, max_context_len, embed_dim]
            neg_masked_embeds = neg_context_embeds * negative_masks.unsqueeze(-1)
            neg_context_lengths = negative_masks.sum(dim=1, keepdim=True)
            neg_context_lengths = torch.clamp(neg_context_lengths, min=1)
            
            avg_neg_context_embeds = neg_masked_embeds.sum(dim=1) / neg_context_lengths  # [num_negatives, embed_dim]
            
            # Get negative word embeddings
            neg_out_embeds = self.embeddings.out_embeddings(negative_words)  # [num_negatives, embed_dim]
            
            # Ensure dimensions match for element-wise multiplication
            assert avg_neg_context_embeds.size() == neg_out_embeds.size(), f"Context embeds {avg_neg_context_embeds.size()} and neg embeds {neg_out_embeds.size()} must match"
            
            # Negative scores (we want these to be low)
            negative_scores = torch.sum(avg_neg_context_embeds * neg_out_embeds, dim=1)  # [num_negatives]
            negative_loss = F.logsigmoid(-negative_scores).mean()
            
            # Total loss
            total_loss = -(positive_loss + negative_loss)
            
            results.update({
                'negative_loss': negative_loss,
                'negative_scores': negative_scores,
                'total_loss': total_loss
            })
        else:
            # Without negative sampling, use full softmax
            all_out_embeds = self.embeddings.out_embeddings.weight  # [vocab_size, embed_dim]
            logits = torch.matmul(avg_context_embeds, all_out_embeds.t())  # [batch_size, vocab_size]
            
            softmax_loss = F.cross_entropy(logits, center_words)
            results.update({
                'softmax_loss': softmax_loss,
                'logits': logits,
                'total_loss': softmax_loss
            })
        
        return results  # [batch_size, vocab_size]
            
        softmax_loss = F.cross_entropy(logits, center_words)
        results.update({
                'softmax_loss': softmax_loss,
                'logits': logits,
                'total_loss': softmax_loss
            })
        
        return results  # [batch_size, vocab_size]
            
        softmax_loss = F.cross_entropy(logits, center_words)
        results.update({
                'softmax_loss': softmax_loss,
                'logits': logits,
                'total_loss': softmax_loss
            })
        
        return results
    
    def predict_center(self, context_words: torch.Tensor, 
                      context_mask: torch.Tensor, 
                      top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict most likely center word for given context words.
        
        Args:
            context_words: Context word indices [1, max_context_len]
            context_mask: Mask for valid context words [1, max_context_len]
            top_k: Number of top predictions to return
            
        Returns:
            Tuple of (top_k_indices, top_k_scores)
        """
        self.eval()
        with torch.no_grad():
            # Get average context embedding
            context_embeds = self.embeddings.in_embeddings(context_words)  # [1, max_context_len, embed_dim]
            masked_embeds = context_embeds * context_mask.unsqueeze(-1)
            context_length = context_mask.sum(dim=1, keepdim=True)
            context_length = torch.clamp(context_length, min=1)
            
            avg_context_embed = masked_embeds.sum(dim=1) / context_length  # [1, embed_dim]
            
            # Compute similarity scores
            all_out_embeds = self.embeddings.out_embeddings.weight  # [vocab_size, embed_dim]
            scores = torch.matmul(avg_context_embed, all_out_embeds.t()).squeeze()  # [vocab_size]
            
            # Get top-k predictions
            top_scores, top_indices = torch.topk(scores, top_k)
            
        return top_indices, top_scores
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Word2VecModel(nn.Module):
    """
    Unified Word2Vec model that can handle both Skip-gram and CBOW architectures.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, model_type: str = 'skipgram'):
        """
        Initialize Word2Vec model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            model_type: Either 'skipgram' or 'cbow'
        """
        super().__init__()
        
        if model_type.lower() not in ['skipgram', 'cbow']:
            raise ValueError("model_type must be 'skipgram' or 'cbow'")
        
        self.model_type = model_type.lower()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Initialize the appropriate model
        if self.model_type == 'skipgram':
            self.model = SkipGramModel(vocab_size, embedding_dim)
        else:
            self.model = CBOWModel(vocab_size, embedding_dim)
    
    def forward(self, *args, **kwargs):
        """Forward pass - delegates to the specific model."""
        return self.model(*args, **kwargs)
    
    def get_word_embeddings(self) -> torch.Tensor:
        """Get the learned word embeddings."""
        return self.model.embeddings.get_word_embeddings()
    
    def get_word_embedding(self, word_id: int) -> torch.Tensor:
        """Get embedding for a specific word."""
        return self.model.embeddings.get_word_embedding(word_id)
    
    def save_embeddings(self, filepath: str, vocab: 'SimpleVocabulary'):
        """
        Save embeddings in a readable format.
        
        Args:
            filepath: Path to save embeddings
            vocab: Vocabulary instance for word mapping
        """
        embeddings = self.get_word_embeddings().cpu().numpy()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{self.vocab_size} {self.embedding_dim}\n")
            
            for word_id in range(self.vocab_size):
                word = vocab.get_word(word_id)
                embedding_str = " ".join(f"{x:.6f}" for x in embeddings[word_id])
                f.write(f"{word} {embedding_str}\n")
        
        print(f"Embeddings saved to {filepath}")
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return self.model.count_parameters()


# =============================================================================
# UTILITY FUNCTIONS FOR MODEL ANALYSIS
# =============================================================================

def print_model_info(model: Word2VecModel) -> None:
    """
    Print comprehensive model information.
    
    Args:
        model: Word2VecModel instance
    """
    print(f"\n{'='*60}")
    print(f"WORD2VEC MODEL INFORMATION")
    print(f"{'='*60}")
    print(f"Model type: {model.model_type.upper()}")
    print(f"Vocabulary size: {model.vocab_size:,}")
    print(f"Embedding dimension: {model.embedding_dim}")
    print(f"Total parameters: {model.count_parameters():,}")
    
    # Memory estimation
    param_memory = model.count_parameters() * 4 / (1024**2)  # Assuming float32
    print(f"Estimated memory: {param_memory:.1f} MB")
    
    # Model architecture details
    print(f"\nModel Architecture:")
    print(f"  - Input embeddings: {model.vocab_size:,} × {model.embedding_dim}")
    print(f"  - Output embeddings: {model.vocab_size:,} × {model.embedding_dim}")
    print(f"  - Training method: {'Negative sampling' if hasattr(model, 'negative_sampler') else 'Full softmax'}")
    
    print("-" * 60)


def demonstrate_model_prediction(model: Word2VecModel, vocab: 'SimpleVocabulary', 
                               demo_words: list, top_k: int = 5) -> None:
    """
    Demonstrate model predictions for given words.
    
    Args:
        model: Trained Word2VecModel
        vocab: Vocabulary instance
        demo_words: List of words to demonstrate
        top_k: Number of top predictions to show
    """
    print(f"\n{'='*60}")
    print(f"MODEL PREDICTION DEMONSTRATION")
    print(f"{'='*60}")
    print(f"Model: {model.model_type.upper()}")
    print(f"Showing top {top_k} predictions for each word")
    print("-" * 60)
    
    model.eval()
    
    for word in demo_words:
        word_id = vocab.get_word_id(word)
        actual_word = vocab.get_word(word_id)
        
        print(f"\nInput word: '{actual_word}' (ID: {word_id})")
        
        if word != actual_word:
            print(f"  Note: '{word}' not in vocabulary, using '{actual_word}'")
        
        try:
            if model.model_type == 'skipgram':
                # Predict context words from center word
                word_tensor = torch.tensor([word_id], dtype=torch.long)
                top_indices, top_scores = model.model.predict_context(word_tensor, top_k=top_k)
                
                print(f"  Predicted context words:")
                for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
                    pred_word = vocab.get_word(idx.item())
                    print(f"    {i+1}. {pred_word} (score: {score.item():.3f})")
            
            elif model.model_type == 'cbow':
                # For CBOW, we need context words to predict center
                # Use the word itself as context for demonstration
                context_tensor = torch.tensor([[word_id]], dtype=torch.long)
                mask_tensor = torch.tensor([[1.0]], dtype=torch.float)
                
                top_indices, top_scores = model.model.predict_center(context_tensor, mask_tensor, top_k=top_k)
                
                print(f"  Predicted center words (using '{actual_word}' as context):")
                for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
                    pred_word = vocab.get_word(idx.item())
                    print(f"    {i+1}. {pred_word} (score: {score.item():.3f})")
        
        except Exception as e:
            print(f"  Error making prediction: {e}")
        
        print("-" * 40)


def analyze_embedding_similarity(model: Word2VecModel, vocab: 'SimpleVocabulary',
                               target_word: str, top_k: int = 10) -> None:
    """
    Analyze word similarities using cosine similarity of embeddings.
    
    Args:
        model: Trained Word2VecModel
        vocab: Vocabulary instance
        target_word: Word to find similarities for
        top_k: Number of most similar words to show
    """
    print(f"\n{'='*60}")
    print(f"EMBEDDING SIMILARITY ANALYSIS")
    print(f"{'='*60}")
    
    target_id = vocab.get_word_id(target_word)
    actual_word = vocab.get_word(target_id)
    
    print(f"Target word: '{actual_word}' (ID: {target_id})")
    
    if target_word != actual_word:
        print(f"Note: '{target_word}' not in vocabulary, using '{actual_word}'")
    
    # Get all embeddings
    all_embeddings = model.get_word_embeddings()  # [vocab_size, embed_dim]
    target_embedding = all_embeddings[target_id].unsqueeze(0)  # [1, embed_dim]
    
    # Compute cosine similarities
    similarities = F.cosine_similarity(target_embedding, all_embeddings, dim=1)  # [vocab_size]
    
    # Get top-k most similar words (excluding the target word itself)
    similarities[target_id] = -1  # Exclude self
    top_scores, top_indices = torch.topk(similarities, top_k)
    
    print(f"\nTop {top_k} most similar words:")
    print("-" * 40)
    print(f"{'Rank':<6} {'Word':<15} {'Similarity':<12}")
    print("-" * 40)
    
    for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
        similar_word = vocab.get_word(idx.item())
        print(f"{i+1:<6} {similar_word:<15} {score.item():<12.3f}")
    
    print("-" * 60)


def compare_embeddings_before_after_training(vocab_size: int, embedding_dim: int, 
                                           model_type: str = 'skipgram') -> None:
    """
    Compare embedding statistics before and after initialization.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embeddings
        model_type: Model type
    """
    print(f"\n{'='*60}")
    print(f"EMBEDDING INITIALIZATION ANALYSIS")
    print(f"{'='*60}")
    
    # Create model
    model = Word2VecModel(vocab_size, embedding_dim, model_type)
    embeddings = model.get_word_embeddings()
    
    # Analyze embedding statistics
    mean_val = embeddings.mean().item()
    std_val = embeddings.std().item()
    min_val = embeddings.min().item()
    max_val = embeddings.max().item()
    
    print(f"Embedding matrix statistics:")
    print(f"  - Shape: {embeddings.shape}")
    print(f"  - Mean: {mean_val:.6f}")
    print(f"  - Std: {std_val:.6f}")
    print(f"  - Min: {min_val:.6f}")
    print(f"  - Max: {max_val:.6f}")
    print(f"  - Initialization range: ±{0.5 / embedding_dim:.6f}")
    
    # Check if initialization is reasonable
    expected_std = np.sqrt(1.0 / (3 * embedding_dim))  # Rough estimate for uniform distribution
    print(f"\nInitialization quality:")
    print(f"  - Expected std (rough): {expected_std:.6f}")
    print(f"  - Actual std: {std_val:.6f}")
    print(f"  - Ratio: {std_val / expected_std:.2f}")
    
    print("-" * 60)


# =============================================================================
# INTEGRATION WITH EXISTING PIPELINE
# =============================================================================

def create_model_from_pipeline(pipeline: 'Word2VecDataPipeline', 
                              embedding_dim: int = 100) -> Word2VecModel:
    """
    Create a Word2Vec model from an existing data pipeline.
    
    Args:
        pipeline: Word2VecDataPipeline instance (must be built)
        embedding_dim: Dimension of word embeddings
        
    Returns:
        Initialized Word2VecModel
    """
    if not pipeline.vocabulary or not pipeline.vocabulary.is_built:
        raise ValueError("Pipeline vocabulary must be built first")
    
    if not pipeline.dataset:
        raise ValueError("Pipeline dataset must be created first")
    
    vocab_size = pipeline.vocabulary.vocab_size
    model_type = pipeline.dataset.model_type
    
    print(f"\n{'='*60}")
    print(f"CREATING MODEL FROM PIPELINE")
    print(f"{'='*60}")
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Model type: {model_type.upper()}")
    print(f"Embedding dimension: {embedding_dim}")
    print("-" * 60)
    
    # Create model
    model = Word2VecModel(vocab_size, embedding_dim, model_type)
    
    print(f"✓ Model created successfully!")
    print(f"✓ Ready for training with pipeline DataLoader")
    
    return model


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Word2Vec Model Architecture...")
    
    # Test parameters
    vocab_size = 1000
    embedding_dim = 100
    
    # Test both model types
    for model_type in ['skipgram', 'cbow']:
        print(f"\n{'#'*80}")
        print(f"TESTING {model_type.upper()} MODEL")
        print(f"{'#'*80}")
        
        # Create model
        model = Word2VecModel(vocab_size, embedding_dim, model_type)
        print_model_info(model)
        
        # Test forward pass
        print(f"\nTesting forward pass...")
        model.train()
        
        if model_type == 'skipgram':
            # Test Skip-gram forward pass
            center_words = torch.randint(0, vocab_size, (8,))
            context_words = torch.randint(0, vocab_size, (8,))
            
            # Without negative sampling
            results = model(center_words, context_words)
            print(f"  Without negative sampling:")
            print(f"    Positive loss: {results['positive_loss'].item():.4f}")
            print(f"    Total loss: {results['total_loss'].item():.4f}")
            
            # With negative sampling
            negative_words = torch.randint(0, vocab_size, (40,))  # 5 negatives per positive
            negative_centers = center_words.repeat_interleave(5)
            
            results = model(center_words, context_words, negative_words, negative_centers)
            print(f"  With negative sampling:")
            print(f"    Positive loss: {results['positive_loss'].item():.4f}")
            print(f"    Negative loss: {results['negative_loss'].item():.4f}")
            print(f"    Total loss: {results['total_loss'].item():.4f}")
        
        else:  # CBOW
            # Test CBOW forward pass
            context_words = torch.randint(0, vocab_size, (8, 5))  # batch_size=8, max_context=5
            context_mask = torch.rand(8, 5) > 0.2  # Random mask
            center_words = torch.randint(0, vocab_size, (8,))
            
            # Without negative sampling
            results = model(context_words, context_mask.float(), center_words)
            print(f"  Without negative sampling:")
            print(f"    Positive loss: {results['positive_loss'].item():.4f}")
            print(f"    Total loss: {results['total_loss'].item():.4f}")
            
            # With negative sampling
            negative_words = torch.randint(0, vocab_size, (40,))
            negative_contexts = context_words.repeat_interleave(5, dim=0)
            negative_masks = context_mask.float().repeat_interleave(5, dim=0)
            
            results = model(context_words, context_mask.float(), center_words,
                          negative_words, negative_contexts, negative_masks)
            # print(f"  With negative sampling:")
