"""
Simple Word2Vec Training - Educational Implementation
Step 7: Training Loop and Model Optimization

This module provides a complete training framework for Word2Vec models:
- Training loop with progress tracking
- Learning rate scheduling
- Loss monitoring and visualization
- Model evaluation and similarity analysis
- Integration with the complete pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np

# Import our previous modules
from word2vec_embedding import Word2VecModel, print_model_info, analyze_embedding_similarity
from vocab_builder import SimpleVocabulary


class Word2VecTrainer:
    """
    Comprehensive trainer for Word2Vec models.
    Handles both Skip-gram and CBOW training with educational transparency.
    """
    
    def __init__(self, 
                 model: Word2VecModel,
                 vocab: SimpleVocabulary,
                 learning_rate: float = 0.025,
                 min_learning_rate: float = 0.0001,
                 weight_decay: float = 0.0):
        """
        Initialize trainer.
        
        Args:
            model: Word2VecModel instance
            vocab: SimpleVocabulary instance
            learning_rate: Initial learning rate
            min_learning_rate: Minimum learning rate for decay
            weight_decay: L2 regularization weight
        """
        self.model = model
        self.vocab = vocab
        self.initial_lr = learning_rate
        self.min_lr = min_learning_rate
        self.weight_decay = weight_decay
        
        # Setup optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.training_history = defaultdict(list)
        self.best_loss = float('inf')
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"\n{'='*60}")
        print(f"WORD2VEC TRAINER INITIALIZED")
        print(f"{'='*60}")
        print(f"Model: {self.model.model_type.upper()}")
        print(f"Device: {self.device}")
        print(f"Initial learning rate: {learning_rate}")
        print(f"Min learning rate: {min_learning_rate}")
        print(f"Weight decay: {weight_decay}")
        print(f"Optimizer: {type(self.optimizer).__name__}")
        print("-" * 60)
    
    def _update_learning_rate(self, epoch: int, total_epochs: int) -> None:
        """
        Update learning rate using linear decay.
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
        """
        # Linear decay from initial_lr to min_lr
        progress = epoch / total_epochs
        lr = self.initial_lr * (1 - progress) + self.min_lr * progress
        lr = max(lr, self.min_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def train_epoch(self, dataloader: DataLoader, epoch: int, total_epochs: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training DataLoader
            epoch: Current epoch number
            total_epochs: Total number of epochs
            
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        epoch_losses = defaultdict(list)
        
        # Update learning rate
        self._update_learning_rate(epoch, total_epochs)
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Training loop
        total_batches = len(dataloader)
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = [tensor.to(self.device) if isinstance(tensor, torch.Tensor) else tensor for tensor in batch]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.model.model_type == 'skipgram':
                results = self._forward_skipgram(batch)
            else:  # cbow
                results = self._forward_cbow(batch)
            
            # Backward pass
            loss = results['total_loss']
            loss.backward()
            
            # Gradient clipping (optional, helps with stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Record losses
            for key, value in results.items():
                if 'loss' in key and isinstance(value, torch.Tensor):
                    epoch_losses[key].append(value.item())
            
            # Progress reporting
            if batch_idx % max(1, total_batches // 10) == 0 or batch_idx == total_batches - 1:
                progress = (batch_idx + 1) / total_batches * 100
                avg_loss = np.mean(epoch_losses['total_loss'][-10:])  # Last 10 batches
                
                print(f"  Epoch {epoch+1}/{total_epochs} [{progress:5.1f}%] "
                      f"Loss: {avg_loss:.4f} LR: {current_lr:.6f}")
            
            self.global_step += 1
        
        # Calculate epoch averages
        epoch_metrics = {}
        for key, values in epoch_losses.items():
            epoch_metrics[f'avg_{key}'] = np.mean(values)
        
        epoch_metrics['learning_rate'] = current_lr
        epoch_metrics['epoch_time'] = time.time() - start_time
        
        return epoch_metrics
    
    def _forward_skipgram(self, batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Handle Skip-gram forward pass."""
        if len(batch) == 2:
            # Without negative sampling
            center_words, context_words = batch
            return self.model(center_words, context_words)
        elif len(batch) == 4:
            # With negative sampling
            center_words, context_words, neg_words, neg_centers = batch
            return self.model(center_words, context_words, neg_words, neg_centers)
        else:
            raise ValueError(f"Unexpected batch format for Skip-gram: {len(batch)} tensors")
    
    def _forward_cbow(self, batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Handle CBOW forward pass."""
        if len(batch) == 3:
            # Without negative sampling
            context_words, context_mask, center_words = batch
            return self.model(context_words, context_mask, center_words)
        elif len(batch) == 6:
            # With negative sampling
            context_words, context_mask, center_words, neg_words, neg_contexts, neg_masks = batch
            return self.model(context_words, context_mask, center_words, 
                            neg_words, neg_contexts, neg_masks)
        else:
            raise ValueError(f"Unexpected batch format for CBOW: {len(batch)} tensors")
    
    def train(self, 
              dataloader: DataLoader, 
              epochs: int = 5,
              eval_words: Optional[List[str]] = None,
              eval_interval: int = 1,
              save_checkpoints: bool = False,
              checkpoint_dir: str = "checkpoints") -> Dict[str, List[float]]:
        """
        Train the Word2Vec model.
        
        Args:
            dataloader: Training DataLoader
            epochs: Number of training epochs
            eval_words: Words to evaluate similarity for during training
            eval_interval: Evaluate every N epochs
            save_checkpoints: Whether to save model checkpoints
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*80}")
        print(f"STARTING WORD2VEC TRAINING")
        print(f"{'='*80}")
        print(f"Model: {self.model.model_type.upper()}")
        print(f"Epochs: {epochs}")
        print(f"Batches per epoch: {len(dataloader):,}")
        print(f"Total training steps: {epochs * len(dataloader):,}")
        print(f"Evaluation words: {eval_words or 'None'}")
        print("-" * 80)
        
        # Create checkpoint directory
        if save_checkpoints:
            import os
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        training_start = time.time()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Train one epoch
            epoch_metrics = self.train_epoch(dataloader, epoch, epochs)
            
            # Update training history
            for key, value in epoch_metrics.items():
                self.training_history[key].append(value)
            
            # Print epoch summary
            print(f"  Epoch {epoch + 1} Summary:")
            print(f"    Average loss: {epoch_metrics['avg_total_loss']:.4f}")
            print(f"    Learning rate: {epoch_metrics['learning_rate']:.6f}")
            print(f"    Epoch time: {epoch_metrics['epoch_time']:.1f}s")
            
            # Evaluation
            if eval_words and (epoch + 1) % eval_interval == 0:
                print(f"\n  Evaluation at epoch {epoch + 1}:")
                self._evaluate_similarities(eval_words)
            
            # Save checkpoint
            if save_checkpoints and (epoch + 1) % max(1, epochs // 5) == 0:
                checkpoint_path = f"{checkpoint_dir}/word2vec_epoch_{epoch+1}.pt"
                self._save_checkpoint(checkpoint_path, epoch + 1, epoch_metrics)
            
            # Update best loss
            current_loss = epoch_metrics['avg_total_loss']
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                print(f"    ✓ New best loss: {self.best_loss:.4f}")
            
            self.current_epoch = epoch + 1
        
        total_time = time.time() - training_start
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETED")
        print(f"{'='*80}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Final learning rate: {self.training_history['learning_rate'][-1]:.6f}")
        print(f"Total steps: {self.global_step:,}")
        print("-" * 80)
        
        # Final evaluation
        if eval_words:
            print(f"\nFinal Model Evaluation:")
            self._evaluate_similarities(eval_words)
        
        return dict(self.training_history)
    
    def _evaluate_similarities(self, eval_words: List[str]) -> None:
        """Evaluate word similarities during training."""
        self.model.eval()
        with torch.no_grad():
            for word in eval_words[:3]:  # Limit to first 3 words
                print(f"    '{word}' similar words:")
                try:
                    word_id = self.vocab.get_word_id(word)
                    actual_word = self.vocab.get_word(word_id)
                    
                    # Get embeddings and compute similarities
                    all_embeddings = self.model.get_word_embeddings()
                    target_embedding = all_embeddings[word_id].unsqueeze(0)
                    
                    similarities = torch.nn.functional.cosine_similarity(
                        target_embedding, all_embeddings, dim=1
                    )
                    similarities[word_id] = -1  # Exclude self
                    
                    top_scores, top_indices = torch.topk(similarities, 3)
                    
                    similar_words = []
                    for idx, score in zip(top_indices, top_scores):
                        similar_word = self.vocab.get_word(idx.item())
                        similar_words.append(f"{similar_word}({score.item():.2f})")
                    
                    print(f"      {', '.join(similar_words)}")
                    
                except Exception as e:
                    print(f"      Error: {e}")
        
        self.model.train()
    
    def _save_checkpoint(self, filepath: str, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': dict(self.training_history),
            'metrics': metrics,
            'vocab_size': self.vocab.vocab_size,
            'embedding_dim': self.model.embedding_dim,
            'model_type': self.model.model_type
        }
        
        torch.save(checkpoint, filepath)
        print(f"    Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = defaultdict(list, checkpoint['training_history'])
        self.current_epoch = checkpoint['epoch']
        
        print(f"Checkpoint loaded from {filepath} (epoch {self.current_epoch})")


# =============================================================================
# EVALUATION AND ANALYSIS FUNCTIONS
# =============================================================================

def evaluate_word_analogies(model: Word2VecModel, vocab: SimpleVocabulary,
                          analogies: List[Tuple[str, str, str, str]]) -> float:
    """
    Evaluate model on word analogy tasks.
    
    Args:
        model: Trained Word2VecModel
        vocab: Vocabulary instance
        analogies: List of (word1, word2, word3, expected_word4) tuples
        
    Returns:
        Accuracy on analogy tasks
    """
    print(f"\n{'='*60}")
    print(f"WORD ANALOGY EVALUATION")
    print(f"{'='*60}")
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        embeddings = model.get_word_embeddings()
        
        for word1, word2, word3, expected in analogies:
            try:
                # Get word IDs
                id1 = vocab.get_word_id(word1)
                id2 = vocab.get_word_id(word2)
                id3 = vocab.get_word_id(word3)
                expected_id = vocab.get_word_id(expected)
                
                # Skip if any word is unknown
                if any(vocab.get_word(id_) == '<UNK>' for id_ in [id1, id2, id3, expected_id]):
                    continue
                
                # Compute analogy: word1 - word2 + word3 ≈ expected
                analogy_vector = embeddings[id1] - embeddings[id2] + embeddings[id3]
                
                # Find most similar word
                similarities = torch.nn.functional.cosine_similarity(
                    analogy_vector.unsqueeze(0), embeddings, dim=1
                )
                
                # Exclude the input words
                similarities[id1] = -1
                similarities[id2] = -1
                similarities[id3] = -1
                
                predicted_id = similarities.argmax().item()
                
                if predicted_id == expected_id:
                    correct += 1
                    result = "✓"
                else:
                    result = "✗"
                    predicted_word = vocab.get_word(predicted_id)
                    print(f"  {word1}:{word2}::{word3}:{expected} → {predicted_word} {result}")
                
                total += 1
                
            except Exception as e:
                print(f"  Error with {word1}:{word2}::{word3}:{expected} - {e}")
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"\nAnalogy Accuracy: {correct}/{total} = {accuracy:.1%}")
    print("-" * 60)
    
    return accuracy


def plot_training_history(training_history: Dict[str, List[float]]) -> None:
    """
    Plot training history (simplified version for educational purposes).
    
    Args:
        training_history: Dictionary with training metrics
    """
    print(f"\n{'='*60}")
    print(f"TRAINING HISTORY SUMMARY")
    print(f"{'='*60}")
    
    epochs = len(training_history['avg_total_loss'])
    
    print(f"Training Progress:")
    print(f"{'Epoch':<8} {'Loss':<10} {'LR':<12} {'Time(s)':<10}")
    print("-" * 45)
    
    for i in range(epochs):
        loss = training_history['avg_total_loss'][i]
        lr = training_history['learning_rate'][i]
        time_taken = training_history['epoch_time'][i]
        
        print(f"{i+1:<8} {loss:<10.4f} {lr:<12.6f} {time_taken:<10.1f}")
    
    # Show improvement
    if epochs > 1:
        initial_loss = training_history['avg_total_loss'][0]
        final_loss = training_history['avg_total_loss'][-1]
        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        
        print(f"\nTraining Summary:")
        print(f"  Initial loss: {initial_loss:.4f}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Improvement: {improvement:.1f}%")
    
    print("-" * 60)


def comprehensive_model_evaluation(model: Word2VecModel, vocab: SimpleVocabulary,
                                 eval_words: List[str] = None) -> None:
    """
    Comprehensive evaluation of trained model.
    
    Args:
        model: Trained Word2VecModel
        vocab: Vocabulary instance
        eval_words: Words to analyze in detail
    """
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE MODEL EVALUATION")
    print(f"{'='*80}")
    
    # Default evaluation words if none provided
    if eval_words is None:
        eval_words = ['the', 'word', 'good', 'learning', 'model']
        # Filter to words that exist in vocab
        eval_words = [word for word in eval_words if vocab.get_word_id(word) != vocab.get_word_id('<UNK>')]
    
    model.eval()
    
    # 1. Model Statistics
    print_model_info(model)
    
    # 2. Embedding Analysis
    embeddings = model.get_word_embeddings()
    print(f"\nEmbedding Statistics:")
    print(f"  Mean: {embeddings.mean().item():.6f}")
    print(f"  Std: {embeddings.std().item():.6f}")
    print(f"  Min: {embeddings.min().item():.6f}")
    print(f"  Max: {embeddings.max().item():.6f}")
    
    # 3. Word Similarity Analysis
    for word in eval_words[:3]:  # Analyze first 3 words
        analyze_embedding_similarity(model, vocab, word, top_k=5)
    
    # 4. Sample Analogies (if enough words available)
    if len(eval_words) >= 4:
        sample_analogies = [
            (eval_words[0], eval_words[1], eval_words[2], eval_words[3])
        ]
        evaluate_word_analogies(model, vocab, sample_analogies)
    
    print(f"\n{'='*80}")
    print("Model evaluation complete!")
    print(f"{'='*80}")


# =============================================================================
# INTEGRATION WITH COMPLETE PIPELINE
# =============================================================================

def train_word2vec_from_pipeline(pipeline: 'Word2VecDataPipeline',
                                embedding_dim: int = 100,
                                epochs: int = 5,
                                learning_rate: float = 0.025,
                                eval_words: List[str] = None) -> Tuple[Word2VecModel, Dict[str, List[float]]]:
    """
    Train Word2Vec model from complete data pipeline.
    
    Args:
        pipeline: Completed Word2VecDataPipeline
        embedding_dim: Embedding dimension
        epochs: Training epochs
        learning_rate: Learning rate
        eval_words: Words for evaluation
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    print(f"\n{'='*100}")
    print(f"COMPLETE WORD2VEC TRAINING FROM PIPELINE")
    print(f"{'='*100}")
    
    # Validate pipeline
    if not pipeline.vocabulary or not pipeline.vocabulary.is_built:
        raise ValueError("Pipeline vocabulary must be built")
    if not pipeline.dataset:
        raise ValueError("Pipeline dataset must be created")
    if not pipeline.dataloader:
        raise ValueError("Pipeline dataloader must be created")
    
    # Create model
    vocab_size = pipeline.vocabulary.vocab_size
    model_type = pipeline.dataset.model_type
    
    print(f"Creating {model_type.upper()} model...")
    model = Word2VecModel(vocab_size, embedding_dim, model_type)
    
    # Create trainer
    trainer = Word2VecTrainer(
        model=model,
        vocab=pipeline.vocabulary,
        learning_rate=learning_rate
    )
    
    # Setup evaluation words
    if eval_words is None:
        # Use most frequent words from vocabulary
        frequent_words = []
        for word, freq in sorted(pipeline.vocabulary.word_freq.items(), 
                               key=lambda x: x[1], reverse=True):
            if word not in pipeline.vocabulary.special_tokens and len(frequent_words) < 10:
                frequent_words.append(word)
        eval_words = frequent_words[:5]
    
    print(f"Evaluation words: {eval_words}")
    
    # Train model
    training_history = trainer.train(
        dataloader=pipeline.dataloader,
        epochs=epochs,
        eval_words=eval_words,
        eval_interval=max(1, epochs // 3)
    )
    
    # Final evaluation
    comprehensive_model_evaluation(model, pipeline.vocabulary, eval_words)
    
    # Plot training history
    plot_training_history(training_history)
    
    print(f"\n{'='*100}")
    print("COMPLETE WORD2VEC TRAINING FINISHED")
    print(f"{'='*100}")
    print("Results:")
    print(f"✓ Model trained successfully")
    print(f"✓ Final loss: {training_history['avg_total_loss'][-1]:.4f}")
    print(f"✓ Embeddings ready for use")
    print(f"✓ Model evaluation completed")
    print("="*100)
    
    return model, training_history


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Word2Vec Trainer...")
    
    # This would typically be used with the complete pipeline:
    # 
    # # 1. Create and run pipeline
    # config = {'min_word_freq': 5, 'window_size': 5, 'use_negative_sampling': True}
    # pipeline = Word2VecDataPipeline(config)
    # dataloader = pipeline.run_complete_pipeline("text_data.txt", "skipgram", batch_size=32)
    # 
    # # 2. Train model
    # model, history = train_word2vec_from_pipeline(
    #     pipeline=pipeline,
    #     embedding_dim=100,
    #     epochs=10,
    #     learning_rate=0.025
    # )
    # 
    # # 3. Save embeddings
    # model.save_embeddings("word_embeddings.txt", pipeline.vocabulary)
    
    print("Training module ready!")
    print("Use train_word2vec_from_pipeline() with your data pipeline for complete training.")