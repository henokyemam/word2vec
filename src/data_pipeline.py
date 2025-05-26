"""
Complete Word2Vec Data Processing Pipeline - Educational Implementation
Final Integration: From Raw Text to Training-Ready DataLoader

This module demonstrates the complete pipeline:
Raw Text → Tokenization → Vocabulary → Subsampling → Negative Sampling → Dataset → DataLoader
"""

import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Union
from pathlib import Path

# Import all our components
from text_processor import SimpleTextProcessor, load_sample_sentences, print_sample_sentences
from vocab_builder import SimpleVocabulary, print_vocabulary_stats, print_frequency_distribution
from sub_sampler import SimpleSubSampler, print_subsampling_demo, convert_sentences_to_ids
from neg_sampler import SimpleNegativeSampler, print_sampling_distribution
from simple_word2vec import (SimpleWord2VecDataset, simple_collate_skipgram, simple_collate_cbow,
                                   print_training_pair_examples, analyze_dataset_statistics, demonstrate_batching)


class Word2VecDataPipeline:
    """
    Complete Word2Vec data processing pipeline.
    Orchestrates all components for educational transparency.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Dictionary containing all pipeline parameters
        """
        self.config = config
        
        # Initialize components
        self.text_processor = SimpleTextProcessor(
            lowercase=config.get('lowercase', True),
            remove_punctuation=config.get('remove_punctuation', True)
        )
        
        # These will be created during pipeline execution
        self.vocabulary = None
        self.subsampler = None
        self.negative_sampler = None
        self.dataset = None
        self.dataloader = None
        
        print(f"\n{'='*80}")
        print(f"WORD2VEC DATA PIPELINE INITIALIZED")
        print(f"{'='*80}")
        print("Configuration:")
        for key, value in config.items():
            print(f"  - {key}: {value}")
        print("-" * 80)
    
    def run_complete_pipeline(self, text_source: Union[str, List[str]], 
                            model_type: str = 'skipgram',
                            batch_size: int = 32) -> DataLoader:
        """
        Run the complete data processing pipeline.
        
        Args:
            text_source: Path to text file or list of sentences
            model_type: 'skipgram' or 'cbow'
            batch_size: Batch size for DataLoader
            
        Returns:
            PyTorch DataLoader ready for training
        """
        print(f"\n{'#'*80}")
        print(f"RUNNING COMPLETE WORD2VEC PIPELINE")
        print(f"{'#'*80}")
        print(f"Target model: {model_type.upper()}")
        print(f"Batch size: {batch_size}")
        
        # Step 1: Load and process text
        tokenized_sentences = self._step1_text_processing(text_source)
        
        # Step 2: Build vocabulary
        self._step2_build_vocabulary(tokenized_sentences)
        
        # Step 3: Setup subsampling (optional)
        id_sentences = self._step3_setup_subsampling(tokenized_sentences)
        
        # Step 4: Setup negative sampling
        self._step4_setup_negative_sampling()
        
        # Step 5: Create dataset
        self._step5_create_dataset(id_sentences, model_type)
        
        # Step 6: Create DataLoader
        self._step6_create_dataloader(batch_size)
        
        # Step 7: Final summary
        self._step7_pipeline_summary()
        
        return self.dataloader
    
    def _step1_text_processing(self, text_source: Union[str, List[str]]) -> List[List[str]]:
        """Step 1: Load and tokenize text."""
        print(f"\n{'='*60}")
        print(f"STEP 1: TEXT PROCESSING")
        print(f"{'='*60}")
        
        # Load sentences
        if isinstance(text_source, str):
            # Load from file
            raw_sentences = load_sample_sentences(
                text_source, 
                num_sentences=self.config.get('max_sentences', 1000)
            )
        else:
            # Use provided sentences
            raw_sentences = text_source
        
        # Display sample sentences
        print_sample_sentences(raw_sentences, max_display=5)
        
        # Tokenize sentences
        tokenized_sentences = []
        for sentence in raw_sentences:
            tokens = self.text_processor.tokenize(sentence)
            if tokens:  # Only keep non-empty sentences
                tokenized_sentences.append(tokens)
        
        print(f"\nTokenization Results:")
        print(f"  - Input sentences: {len(raw_sentences):,}")
        print(f"  - Output sentences: {len(tokenized_sentences):,}")
        print(f"  - Empty sentences removed: {len(raw_sentences) - len(tokenized_sentences):,}")
        
        # Show tokenization examples
        print(f"\nTokenization Examples:")
        for i, (raw, tokens) in enumerate(zip(raw_sentences[:3], tokenized_sentences[:3])):
            print(f"  {i+1}. Raw: {raw}")
            print(f"     Tokens: {tokens}")
        
        return tokenized_sentences
    
    def _step2_build_vocabulary(self, tokenized_sentences: List[List[str]]) -> None:
        """Step 2: Build vocabulary from tokenized sentences."""
        print(f"\n{'='*60}")
        print(f"STEP 2: VOCABULARY BUILDING")
        print(f"{'='*60}")
        
        # Create vocabulary
        self.vocabulary = SimpleVocabulary(
            min_freq=self.config.get('min_word_freq', 5),
            max_size=self.config.get('max_vocab_size', None)
        )
        
        # Build from sentences
        self.vocabulary.build_from_sentences(tokenized_sentences)
        
        # Show statistics
        print_vocabulary_stats(self.vocabulary)
        print_frequency_distribution(self.vocabulary, top_n=15)
    
    def _step3_setup_subsampling(self, tokenized_sentences: List[List[str]]) -> List[List[int]]:
        """Step 3: Convert to IDs and setup subsampling."""
        print(f"\n{'='*60}")
        print(f"STEP 3: SUBSAMPLING SETUP")
        print(f"{'='*60}")
        
        # Convert sentences to word IDs
        id_sentences = convert_sentences_to_ids(tokenized_sentences, self.vocabulary)
        
        # Setup subsampling if enabled
        if self.config.get('use_subsampling', False):
            self.subsampler = SimpleSubSampler(
                vocab=self.vocabulary,
                threshold=self.config.get('subsample_threshold', 1e-3)
            )
            
            # Demonstrate subsampling
            print_subsampling_demo(
                self.subsampler, 
                tokenized_sentences[:3], 
                num_demos=3, 
                num_runs=2
            )
            
            # Apply subsampling to all sentences
            print(f"\nApplying subsampling to all sentences...")
            original_word_count = sum(len(sentence) for sentence in id_sentences)
            id_sentences = self.subsampler.subsample_sentences(id_sentences)
            new_word_count = sum(len(sentence) for sentence in id_sentences)
            
            print(f"  - Original words: {original_word_count:,}")
            print(f"  - After subsampling: {new_word_count:,}")
            print(f"  - Reduction: {((original_word_count - new_word_count) / original_word_count) * 100:.1f}%")
        
        else:
            print("Subsampling disabled. Using all words.")
        
        return id_sentences
    
    def _step4_setup_negative_sampling(self) -> None:
        """Step 4: Setup negative sampling."""
        print(f"\n{'='*60}")
        print(f"STEP 4: NEGATIVE SAMPLING SETUP")
        print(f"{'='*60}")
        
        if self.config.get('use_negative_sampling', True):
            self.negative_sampler = SimpleNegativeSampler(
                vocab=self.vocabulary,
                num_negatives=self.config.get('num_negatives', 5)
            )
            
            # Show sampling distribution
            print_sampling_distribution(self.negative_sampler, top_n=10)
        else:
            print("Negative sampling disabled.")
    
    def _step5_create_dataset(self, id_sentences: List[List[int]], model_type: str) -> None:
        """Step 5: Create Word2Vec dataset."""
        print(f"\n{'='*60}")
        print(f"STEP 5: DATASET CREATION")
        print(f"{'='*60}")
        
        # Filter out empty sentences (from subsampling)
        non_empty_sentences = [sentence for sentence in id_sentences if len(sentence) >= 2]
        
        print(f"Sentence filtering:")
        print(f"  - Input sentences: {len(id_sentences):,}")
        print(f"  - Valid sentences (≥2 words): {len(non_empty_sentences):,}")
        print(f"  - Filtered out: {len(id_sentences) - len(non_empty_sentences):,}")
        
        # Create dataset
        self.dataset = SimpleWord2VecDataset(
            sentences=non_empty_sentences,
            vocab=self.vocabulary,
            model_type=model_type,
            window_size=self.config.get('window_size', 5),
            negative_sampler=self.negative_sampler
        )
        
        # Show examples and statistics
        print_training_pair_examples(self.dataset, num_examples=8)
        analyze_dataset_statistics(self.dataset)
    
    def _step6_create_dataloader(self, batch_size: int) -> None:
        """Step 6: Create PyTorch DataLoader."""
        print(f"\n{'='*60}")
        print(f"STEP 6: DATALOADER CREATION")
        print(f"{'='*60}")
        
        # Choose appropriate collate function
        if self.dataset.model_type == 'skipgram':
            collate_fn = simple_collate_skipgram
        else:
            collate_fn = simple_collate_cbow
        
        # Create DataLoader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=self.config.get('shuffle', True),
            collate_fn=collate_fn,
            num_workers=0  # Keep simple for educational purposes
        )
        
        print(f"DataLoader created:")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Shuffle: {self.config.get('shuffle', True)}")
        print(f"  - Total batches: {len(self.dataloader):,}")
        print(f"  - Collate function: {collate_fn.__name__}")
        
        # Demonstrate batching
        demonstrate_batching(self.dataset, batch_size=min(batch_size, 4))
    
    def _step7_pipeline_summary(self) -> None:
        """Step 7: Display final pipeline summary."""
        print(f"\n{'='*80}")
        print(f"PIPELINE SUMMARY")
        print(f"{'='*80}")
        
        print(f"Data Flow Summary:")
        print(f"  Raw Text → Tokenization → Vocabulary → Subsampling → Dataset → DataLoader")
        print()
        
        print(f"Final Statistics:")
        print(f"  - Vocabulary size: {self.vocabulary.vocab_size:,}")
        print(f"  - Training pairs: {len(self.dataset):,}")
        print(f"  - Batch size: {self.dataloader.batch_size}")
        print(f"  - Total batches: {len(self.dataloader):,}")
        print(f"  - Model type: {self.dataset.model_type.upper()}")
        print(f"  - Window size: {self.dataset.window_size}")
        print(f"  - Negative sampling: {'Yes' if self.negative_sampler else 'No'}")
        if self.negative_sampler:
            print(f"  - Negatives per positive: {self.negative_sampler.num_negatives}")
        print(f"  - Subsampling: {'Yes' if self.subsampler else 'No'}")
        if self.subsampler:
            print(f"  - Subsample threshold: {self.subsampler.threshold}")
        
        print(f"\n✓ Pipeline complete! DataLoader ready for training.")
        print("-" * 80)


def demonstrate_complete_pipeline_with_sample_data():
    """
    Demonstrate the complete pipeline using sample data.
    Perfect for educational purposes and testing.
    """
    print(f"\n{'#'*100}")
    print(f"COMPLETE WORD2VEC PIPELINE DEMONSTRATION")
    print(f"{'#'*100}")
    
    # Sample text data
    sample_text = [
        "The quick brown fox jumps over the lazy dog in the garden.",
        "Word2Vec learns distributed representations of words from large text corpora.",
        "The algorithm uses either Skip-gram or CBOW architecture for training.",
        "Skip-gram predicts context words given a center word.",
        "CBOW predicts the center word given context words.",
        "Negative sampling improves training efficiency significantly.",
        "Subsampling of frequent words helps balance the training data.",
        "The model learns meaningful word embeddings through this process.",
        "These embeddings capture semantic relationships between words.",
        "Similar words appear close together in the embedding space.",
        "The training process involves sliding windows over text.",
        "Each window generates multiple training pairs for the model.",
        "The final embeddings can be used for various NLP tasks.",
        "This implementation provides educational transparency throughout.",
        "Students can see every step of the data processing pipeline."
    ]
    
    # Configuration for different scenarios
    configs = [
        {
            'name': 'Basic Configuration',
            'lowercase': True,
            'remove_punctuation': True,
            'min_word_freq': 2,
            'max_vocab_size': None,
            'use_subsampling': False,
            'use_negative_sampling': True,
            'num_negatives': 3,
            'window_size': 3,
            'shuffle': True
        },
        {
            'name': 'Advanced Configuration',
            'lowercase': True,
            'remove_punctuation': True,
            'min_word_freq': 1,
            'max_vocab_size': 100,
            'use_subsampling': True,
            'subsample_threshold': 1e-3,
            'use_negative_sampling': True,
            'num_negatives': 5,
            'window_size': 5,
            'shuffle': True
        }
    ]
    
    # Test both model types with different configurations
    for config in configs:
        for model_type in ['skipgram', 'cbow']:
            print(f"\n{'='*100}")
            print(f"TESTING: {config['name']} + {model_type.upper()} Model")
            print(f"{'='*100}")
            
            # Create and run pipeline
            pipeline = Word2VecDataPipeline(config)
            dataloader = pipeline.run_complete_pipeline(
                text_source=sample_text,
                model_type=model_type,
                batch_size=8
            )
            
            # Test the DataLoader
            print(f"\nTesting DataLoader:")
            print("-" * 40)
            batch_count = 0
            total_pairs = 0
            
            for batch in dataloader:
                batch_count += 1
                if model_type == 'skipgram':
                    if len(batch) == 2:
                        center, context = batch
                        batch_size = center.size(0)
                    else:
                        center, context, neg, neg_center = batch
                        batch_size = center.size(0)
                else:  # cbow
                    if len(batch) == 3:
                        context, mask, center = batch
                        batch_size = center.size(0)
                    else:
                        context, mask, center, neg, neg_context, neg_mask = batch
                        batch_size = center.size(0)
                
                total_pairs += batch_size
                
                if batch_count <= 2:  # Show first 2 batches
                    print(f"  Batch {batch_count}: {batch_size} pairs")
                    for i, tensor in enumerate(batch):
                        print(f"    Tensor {i+1}: {tensor.shape}")
                
                if batch_count >= 5:  # Limit testing
                    break
            
            print(f"  Total batches tested: {batch_count}")
            print(f"  Total pairs processed: {total_pairs}")
            print(f"  ✓ DataLoader working correctly!")
            
            # Memory and performance info
            dataset_size = len(pipeline.dataset)
            vocab_size = pipeline.vocabulary.vocab_size
            print(f"\nPerformance Info:")
            print(f"  - Dataset size: {dataset_size:,} pairs")
            print(f"  - Vocabulary size: {vocab_size:,}")
            print(f"  - Memory per pair: ~{model_type == 'cbow' and 'variable' or 'fixed'}")


def create_sample_text_file(filepath: str = "sample_text.txt"):
    """
    Create a sample text file for testing the complete pipeline.
    
    Args:
        filepath: Path where to save the sample file
    """
    sample_content = """The Word2Vec algorithm learns distributed representations of words.
                        Skip-gram architecture predicts context words from center words.
                        CBOW architecture predicts center words from context words.
                        Negative sampling improves computational efficiency during training.
                        Subsampling of frequent words helps balance the training data.
                        The model captures semantic relationships between words effectively.
                        Similar words appear close together in the embedding space.
                        These embeddings are useful for many natural language processing tasks.
                        The training process involves sliding windows over text documents.
                        Each window generates multiple positive and negative training examples.
                        The neural network learns to distinguish positive pairs from negative pairs.
                        Gradient descent optimizes the embedding vectors during training.
                        The final word vectors encode rich semantic and syntactic information.
                        This implementation provides educational transparency throughout the process.
                        Students can observe each step of the data processing pipeline clearly."""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(sample_content)
    
    print(f"Sample text file created: {filepath}")
    return filepath


def test_with_text_file():
    """Test the pipeline with a text file input."""
    print(f"\n{'#'*100}")
    print(f"TESTING PIPELINE WITH TEXT FILE")
    print(f"{'#'*100}")
    
    # Create sample file
    filepath = create_sample_text_file()
    
    # Simple configuration
    config = {
        'lowercase': True,
        'remove_punctuation': True,
        'min_word_freq': 1,
        'max_vocab_size': None,
        'use_subsampling': True,
        'subsample_threshold': 1e-3,
        'use_negative_sampling': True,
        'num_negatives': 5,
        'window_size': 5,
        'shuffle': True,
        'max_sentences': 20  # Limit for demonstration
    }
    
    # Test with file input
    pipeline = Word2VecDataPipeline(config)
    dataloader = pipeline.run_complete_pipeline(
        text_source=filepath,
        model_type='skipgram',
        batch_size=16
    )
    
    print(f"\n✓ File-based pipeline test completed successfully!")
    
    # Clean up
    Path(filepath).unlink()
    print(f"Cleaned up sample file: {filepath}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run comprehensive demonstrations
    demonstrate_complete_pipeline_with_sample_data()
    test_with_text_file()
    
    print(f"\n{'='*100}")
    print(f"COMPLETE WORD2VEC PIPELINE DEMONSTRATION FINISHED")
    print(f"{'='*100}")
    print("Key Educational Achievements:")
    print("✓ Complete transparency from raw text to training batches")
    print("✓ Step-by-step pipeline with detailed explanations")
    print("✓ Both Skip-gram and CBOW model support")
    print("✓ Optional subsampling and negative sampling")
    print("✓ Comprehensive statistics and visualizations")
    print("✓ Ready-to-use PyTorch DataLoader output")
    print("✓ Educational code suitable for learning and experimentation")
    print("\nThe pipeline is now complete and ready for Word2Vec model training!")
    print("="*100)

    # config = {'min_word_freq': 5, 'window_size': 5}
    # pipeline = Word2VecDataPipeline(config)
    # dataloader = pipeline.run_complete_pipeline("text.txt", "skipgram", batch_size=32)

#     config = {
#     'use_subsampling': True,
#     'use_negative_sampling': True,
#     'num_negatives': 5
# }
# pipeline = Word2VecDataPipeline(config)
# dataloader = pipeline.run_complete_pipeline(sentence_list, "cbow", batch_size=64)