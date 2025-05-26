"""
Train and Evaluate Word2Vec Model
Complete workflow for training Word2Vec on a training file and evaluating on a separate evaluation file.

Usage:
    python train_and_evaluate_word2vec.py --train_file path/to/train.txt --eval_file path/to/eval.txt
"""

import torch
import numpy as np
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Import our Word2Vec modules
from data_pipeline import Word2VecDataPipeline
from word2vec_embedding import Word2VecModel, analyze_embedding_similarity
from word2vec_trainer import Word2VecTrainer, comprehensive_model_evaluation
from vocab_builder import SimpleVocabulary


class Word2VecExperiment:
    """
    Complete Word2Vec training and evaluation experiment.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize experiment with configuration.
        
        Args:
            config: Dictionary with all experiment parameters
        """
        self.config = config
        self.model = None
        self.vocabulary = None
        self.training_history = None
        self.eval_results = {}
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.get('random_seed', 42))
        np.random.seed(config.get('random_seed', 42))
        
        print(f"\n{'='*80}")
        print(f"WORD2VEC EXPERIMENT INITIALIZED")
        print(f"{'='*80}")
        print("Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print("-" * 80)
    
    def train_model(self, train_file: str) -> Tuple[Word2VecModel, Dict]:
        """
        Train Word2Vec model on training file.
        
        Args:
            train_file: Path to training text file
            
        Returns:
            Tuple of (trained_model, training_history)
        """
        print(f"\n{'='*80}")
        print(f"TRAINING WORD2VEC MODEL")
        print(f"{'='*80}")
        print(f"Training file: {train_file}")
        
        # Step 1: Create data pipeline
        pipeline_config = {
            'lowercase': self.config.get('lowercase', True),
            'remove_punctuation': self.config.get('remove_punctuation', True),
            'min_word_freq': self.config.get('min_word_freq', 5),
            'max_vocab_size': self.config.get('max_vocab_size', None),
            'use_subsampling': self.config.get('use_subsampling', True),
            'subsample_threshold': self.config.get('subsample_threshold', 1e-3),
            'use_negative_sampling': self.config.get('use_negative_sampling', True),
            'num_negatives': self.config.get('num_negatives', 5),
            'window_size': self.config.get('window_size', 5),
            'shuffle': self.config.get('shuffle', True),
            'max_sentences': self.config.get('max_train_sentences', None)
        }
        
        pipeline = Word2VecDataPipeline(pipeline_config)
        dataloader = pipeline.run_complete_pipeline(
            text_source=train_file,
            model_type=self.config.get('model_type', 'skipgram'),
            batch_size=self.config.get('batch_size', 512)
        )
        
        # Store vocabulary for evaluation
        self.vocabulary = pipeline.vocabulary
        
        # Step 2: Create model
        vocab_size = pipeline.vocabulary.vocab_size
        embedding_dim = self.config.get('embedding_dim', 300)
        model_type = self.config.get('model_type', 'skipgram')
        
        print(f"\nCreating {model_type.upper()} model...")
        print(f"  Vocabulary size: {vocab_size:,}")
        print(f"  Embedding dimension: {embedding_dim}")
        
        model = Word2VecModel(vocab_size, embedding_dim, model_type)
        
        # Step 3: Train model
        trainer = Word2VecTrainer(
            model=model,
            vocab=pipeline.vocabulary,
            learning_rate=self.config.get('learning_rate', 0.025),
            min_learning_rate=self.config.get('min_learning_rate', 0.0001),
            weight_decay=self.config.get('weight_decay', 0.0)
        )
        
        # Get evaluation words for training monitoring
        eval_words = self._get_evaluation_words(pipeline.vocabulary)
        
        training_history = trainer.train(
            dataloader=dataloader,
            epochs=self.config.get('epochs', 5),
            eval_words=eval_words,
            eval_interval=self.config.get('eval_interval', 1),
            save_checkpoints=self.config.get('save_checkpoints', False),
            checkpoint_dir=self.config.get('checkpoint_dir', 'checkpoints')
        )
        
        self.model = model
        self.training_history = training_history
        
        print(f"\n✅ Training completed!")
        print(f"Final loss: {training_history['avg_total_loss'][-1]:.4f}")
        
        return model, training_history
    
    def evaluate_model(self, eval_file: str) -> Dict:
        """
        Evaluate trained model on evaluation file.
        
        Args:
            eval_file: Path to evaluation text file
            
        Returns:
            Dictionary with evaluation results
        """
        if self.model is None or self.vocabulary is None:
            raise ValueError("Model must be trained before evaluation")
        
        print(f"\n{'='*80}")
        print(f"EVALUATING WORD2VEC MODEL")
        print(f"{'='*80}")
        print(f"Evaluation file: {eval_file}")
        
        # Load evaluation data
        eval_sentences = self._load_text_file(eval_file)
        print(f"Loaded {len(eval_sentences)} evaluation sentences")
        
        # Process evaluation data with same vocabulary
        eval_results = {}
        
        # 1. Vocabulary coverage analysis
        eval_results['vocabulary_coverage'] = self._evaluate_vocabulary_coverage(eval_sentences)
        
        # 2. Perplexity evaluation (if applicable)
        if self.config.get('compute_perplexity', False):
            eval_results['perplexity'] = self._evaluate_perplexity(eval_sentences)
        
        # 3. Word similarity evaluation
        eval_results['word_similarities'] = self._evaluate_word_similarities()
        
        # 4. Analogy evaluation (if provided)
        if self.config.get('analogy_file'):
            eval_results['analogies'] = self._evaluate_analogies(self.config['analogy_file'])
        
        # 5. Embedding quality metrics
        eval_results['embedding_quality'] = self._evaluate_embedding_quality()
        
        # 6. Most similar words analysis
        eval_results['similarity_examples'] = self._evaluate_similarity_examples()
        
        self.eval_results = eval_results
        
        # Print evaluation summary
        self._print_evaluation_summary(eval_results)
        
        return eval_results
    
    def _load_text_file(self, filepath: str) -> List[str]:
        """Load sentences from text file."""
        sentences = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.append(line)
        return sentences
    
    def _get_evaluation_words(self, vocab: SimpleVocabulary, num_words: int = 10) -> List[str]:
        """Get words for evaluation during training."""
        # Get most frequent words (excluding special tokens)
        frequent_words = []
        for word, freq in sorted(vocab.word_freq.items(), key=lambda x: x[1], reverse=True):
            if word not in vocab.special_tokens and len(frequent_words) < num_words:
                frequent_words.append(word)
        return frequent_words
    
    def _evaluate_vocabulary_coverage(self, eval_sentences: List[str]) -> Dict:
        """Evaluate vocabulary coverage on evaluation data."""
        from text_processor import SimpleTextProcessor
        
        # Process evaluation sentences
        processor = SimpleTextProcessor(
            lowercase=self.config.get('lowercase', True),
            remove_punctuation=self.config.get('remove_punctuation', True)
        )
        
        eval_tokens = []
        for sentence in eval_sentences:
            tokens = processor.tokenize(sentence)
            eval_tokens.extend(tokens)
        
        # Calculate coverage
        total_tokens = len(eval_tokens)
        unique_tokens = set(eval_tokens)
        
        # Check how many tokens are in vocabulary
        known_tokens = 0
        unknown_tokens = []
        
        for token in eval_tokens:
            word_id = self.vocabulary.get_word_id(token)
            if self.vocabulary.get_word(word_id) != '<UNK>':
                known_tokens += 1
            else:
                unknown_tokens.append(token)
        
        # Calculate unique word coverage
        known_unique = 0
        for token in unique_tokens:
            word_id = self.vocabulary.get_word_id(token)
            if self.vocabulary.get_word(word_id) != '<UNK>':
                known_unique += 1
        
        coverage_results = {
            'total_tokens': total_tokens,
            'unique_tokens': len(unique_tokens),
            'known_tokens': known_tokens,
            'token_coverage': (known_tokens / total_tokens) * 100,
            'unique_coverage': (known_unique / len(unique_tokens)) * 100,
            'unknown_sample': unknown_tokens[:20],  # Sample of unknown words
            'oov_rate': ((total_tokens - known_tokens) / total_tokens) * 100
        }
        
        return coverage_results
    
    def _evaluate_word_similarities(self) -> Dict:
        """Evaluate word similarities for common words."""
        similarity_results = {}
        
        # Get some common words to test
        test_words = ['the', 'and', 'of', 'to', 'in', 'for', 'with', 'on', 'by', 'from']
        
        for word in test_words:
            word_id = self.vocabulary.get_word_id(word)
            if self.vocabulary.get_word(word_id) != '<UNK>':
                # Get embeddings
                embeddings = self.model.get_word_embeddings()
                target_embedding = embeddings[word_id].unsqueeze(0)
                
                # Compute similarities
                similarities = torch.nn.functional.cosine_similarity(
                    target_embedding, embeddings, dim=1
                )
                similarities[word_id] = -1  # Exclude self
                
                # Get top similar words
                top_scores, top_indices = torch.topk(similarities, 5)
                similar_words = []
                
                for idx, score in zip(top_indices, top_scores):
                    similar_word = self.vocabulary.get_word(idx.item())
                    similar_words.append((similar_word, score.item()))
                
                similarity_results[word] = similar_words
        
        return similarity_results
    
    def _evaluate_analogies(self, analogy_file: str) -> Dict:
        """Evaluate word analogies if analogy file is provided."""
        # This would implement analogy evaluation
        # For now, return placeholder
        return {'analogy_accuracy': 0.0, 'total_analogies': 0}
    
    def _evaluate_embedding_quality(self) -> Dict:
        """Evaluate general embedding quality metrics."""
        embeddings = self.model.get_word_embeddings()
        
        quality_metrics = {
            'embedding_dim': embeddings.shape[1],
            'vocab_size': embeddings.shape[0],
            'mean_norm': torch.norm(embeddings, dim=1).mean().item(),
            'std_norm': torch.norm(embeddings, dim=1).std().item(),
            'mean_value': embeddings.mean().item(),
            'std_value': embeddings.std().item(),
            'min_value': embeddings.min().item(),
            'max_value': embeddings.max().item()
        }
        
        return quality_metrics
    
    def _evaluate_similarity_examples(self) -> Dict:
        """Get similarity examples for interesting words."""
        examples = {}
        interesting_words = ['king', 'queen', 'man', 'woman', 'good', 'bad', 'big', 'small']
        
        for word in interesting_words:
            word_id = self.vocabulary.get_word_id(word)
            if self.vocabulary.get_word(word_id) != '<UNK>':
                embeddings = self.model.get_word_embeddings()
                target_embedding = embeddings[word_id].unsqueeze(0)
                
                similarities = torch.nn.functional.cosine_similarity(
                    target_embedding, embeddings, dim=1
                )
                similarities[word_id] = -1
                
                top_scores, top_indices = torch.topk(similarities, 3)
                similar_words = []
                
                for idx, score in zip(top_indices, top_scores):
                    similar_word = self.vocabulary.get_word(idx.item())
                    similar_words.append((similar_word, score.item()))
                
                examples[word] = similar_words
        
        return examples
    
    def _print_evaluation_summary(self, eval_results: Dict):
        """Print comprehensive evaluation summary."""
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS SUMMARY")
        print(f"{'='*60}")
        
        # Vocabulary coverage
        if 'vocabulary_coverage' in eval_results:
            cov = eval_results['vocabulary_coverage']
            print(f"\n📊 Vocabulary Coverage:")
            print(f"  Token coverage: {cov['token_coverage']:.1f}%")
            print(f"  Unique word coverage: {cov['unique_coverage']:.1f}%")
            print(f"  Out-of-vocabulary rate: {cov['oov_rate']:.1f}%")
            print(f"  Total evaluation tokens: {cov['total_tokens']:,}")
        
        # Embedding quality
        if 'embedding_quality' in eval_results:
            eq = eval_results['embedding_quality']
            print(f"\n🎯 Embedding Quality:")
            print(f"  Vocabulary size: {eq['vocab_size']:,}")
            print(f"  Embedding dimension: {eq['embedding_dim']}")
            print(f"  Mean embedding norm: {eq['mean_norm']:.3f}")
            print(f"  Embedding value range: [{eq['min_value']:.3f}, {eq['max_value']:.3f}]")
        
        # Word similarities
        if 'word_similarities' in eval_results:
            print(f"\n🔍 Word Similarity Examples:")
            for word, similarities in list(eval_results['word_similarities'].items())[:5]:
                sim_str = ', '.join([f"{w}({s:.2f})" for w, s in similarities[:3]])
                print(f"  '{word}': {sim_str}")
        
        # Similarity examples
        if 'similarity_examples' in eval_results:
            print(f"\n💡 Interesting Similarity Examples:")
            for word, similarities in list(eval_results['similarity_examples'].items())[:3]:
                if similarities:
                    sim_str = ', '.join([f"{w}({s:.2f})" for w, s in similarities])
                    print(f"  '{word}': {sim_str}")
        
        print(f"\n" + "="*60)
    
    def save_results(self, output_dir: str):
        """Save model and evaluation results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n💾 Saving results to {output_dir}")
        
        # Save model
        model_path = output_path / "word2vec_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.vocabulary.vocab_size,
            'embedding_dim': self.model.embedding_dim,
            'model_type': self.model.model_type,
            'config': self.config
        }, model_path)
        print(f"  ✓ Model saved: {model_path}")
        
        # Save vocabulary
        vocab_path = output_path / "vocabulary.json"
        vocab_data = {
            'word2idx': dict(self.vocabulary.word2idx),
            'idx2word': dict(self.vocabulary.idx2word),
            'word_freq': dict(self.vocabulary.word_freq),
            'vocab_size': self.vocabulary.vocab_size,
            'special_tokens': self.vocabulary.special_tokens
        }
        with open(vocab_path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        print(f"  ✓ Vocabulary saved: {vocab_path}")
        
        # Save embeddings in text format
        embeddings_path = output_path / "word_embeddings.txt"
        self.model.save_embeddings(str(embeddings_path), self.vocabulary)
        print(f"  ✓ Embeddings saved: {embeddings_path}")
        
        # Save evaluation results
        if self.eval_results:
            eval_path = output_path / "evaluation_results.json"
            # Convert tensors to Python types for JSON serialization
            serializable_results = self._make_serializable(self.eval_results)
            with open(eval_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"  ✓ Evaluation results saved: {eval_path}")
        
        # Save training history
        if self.training_history:
            history_path = output_path / "training_history.json"
            serializable_history = self._make_serializable(self.training_history)
            with open(history_path, 'w') as f:
                json.dump(serializable_history, f, indent=2)
            print(f"  ✓ Training history saved: {history_path}")
    
    def _make_serializable(self, obj):
        """Convert tensors and other non-serializable objects to Python types."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, (np.ndarray, np.float32, np.float64, np.int32, np.int64)):
            return float(obj) if obj.ndim == 0 else obj.tolist()
        else:
            return obj


def create_default_config() -> Dict:
    """Create default configuration for Word2Vec training."""
    return {
        # Model parameters
        'model_type': 'skipgram',  # 'skipgram' or 'cbow'
        'embedding_dim': 300,
        'window_size': 5,
        'min_word_freq': 5,
        'max_vocab_size': None,
        
        # Training parameters
        'epochs': 5,
        'batch_size': 512,
        'learning_rate': 0.025,
        'min_learning_rate': 0.0001,
        'weight_decay': 0.0,
        
        # Data processing
        'lowercase': True,
        'remove_punctuation': True,
        'use_subsampling': True,
        'subsample_threshold': 1e-3,
        'use_negative_sampling': True,
        'num_negatives': 5,
        'shuffle': True,
        
        # Evaluation
        'compute_perplexity': False,
        'eval_interval': 1,
        
        # I/O
        'save_checkpoints': False,
        'checkpoint_dir': 'checkpoints',
        'random_seed': 42,
        'max_train_sentences': None,  # None for all sentences
    }


def main():
    """Main training and evaluation workflow."""
    parser = argparse.ArgumentParser(description='Train and evaluate Word2Vec model')
    parser.add_argument('--train_file', required=True, help='Path to training text file')
    parser.add_argument('--eval_file', required=True, help='Path to evaluation text file')
    parser.add_argument('--output_dir', default='word2vec_results', help='Output directory for results')
    parser.add_argument('--config_file', help='JSON file with configuration parameters')
    parser.add_argument('--model_type', choices=['skipgram', 'cbow'], default='skipgram', help='Model architecture')
    parser.add_argument('--embedding_dim', type=int, default=300, help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='Learning rate')
    parser.add_argument('--window_size', type=int, default=5, help='Context window size')
    parser.add_argument('--min_word_freq', type=int, default=5, help='Minimum word frequency')
    parser.add_argument('--num_negatives', type=int, default=5, help='Number of negative samples')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_file and Path(args.config_file).exists():
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {args.config_file}")
    else:
        config = create_default_config()
        print("Using default configuration")
    
    # Override config with command line arguments
    config.update({
        'model_type': args.model_type,
        'embedding_dim': args.embedding_dim,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'window_size': args.window_size,
        'min_word_freq': args.min_word_freq,
        'num_negatives': args.num_negatives
    })
    
    # Verify input files exist
    if not Path(args.train_file).exists():
        raise FileNotFoundError(f"Training file not found: {args.train_file}")
    if not Path(args.eval_file).exists():
        raise FileNotFoundError(f"Evaluation file not found: {args.eval_file}")
    
    print(f"🚀 Starting Word2Vec training and evaluation")
    print(f"Training file: {args.train_file}")
    print(f"Evaluation file: {args.eval_file}")
    print(f"Output directory: {args.output_dir}")
    
    # Create experiment
    experiment = Word2VecExperiment(config)
    
    try:
        # Train model
        start_time = time.time()
        model, training_history = experiment.train_model(args.train_file)
        training_time = time.time() - start_time
        
        print(f"\n⏱️  Training completed in {training_time:.1f} seconds")
        
        # Evaluate model
        eval_start = time.time()
        eval_results = experiment.evaluate_model(args.eval_file)
        eval_time = time.time() - eval_start
        
        print(f"\n⏱️  Evaluation completed in {eval_time:.1f} seconds")
        
        # Save results
        experiment.save_results(args.output_dir)
        
        print(f"\n🎉 Word2Vec experiment completed successfully!")
        print(f"Total time: {(time.time() - start_time):.1f} seconds")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())