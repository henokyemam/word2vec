"""
Simple Word2Vec Data Processing - Educational Implementation
Step 4: Negative Sampling
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import Counter
from vocab_builder import SimpleVocabulary


class SimpleNegativeSampler:
    """
    Simple negative sampling implementation for Word2Vec.
    Samples negative examples based on unigram distribution raised to 3/4 power.
    
    From Word2Vec paper:
    - Negative words are sampled according to P(w) ∝ f(w)^(3/4)
    - where f(w) is the frequency of word w
    """
    
    def __init__(self, vocab: SimpleVocabulary, num_negatives: int = 5):
        """
        Initialize negative sampler.
        
        Args:
            vocab: SimpleVocabulary instance
            num_negatives: Number of negative samples per positive example
        """
        if not vocab.is_built:
            raise ValueError("Vocabulary must be built before creating negative sampler")
        
        self.vocab = vocab
        self.num_negatives = num_negatives
        
        # Calculate sampling probabilities
        self.sampling_probs = self._calculate_sampling_probabilities()
        self.word_ids = list(range(vocab.vocab_size))
        
        print(f"\n{'='*60}")
        print(f"NEGATIVE SAMPLER INITIALIZED")
        print(f"{'='*60}")
        print(f"Number of negatives per positive: {self.num_negatives}")
        print(f"Vocabulary size: {vocab.vocab_size}")
        print(f"Sampling distribution: f(w)^0.75 (3/4 power)")
        print("-" * 60)
    
    def _calculate_sampling_probabilities(self) -> np.ndarray:
        """
        Calculate sampling probabilities using 3/4 power of unigram distribution.
        
        Formula: P(w) ∝ f(w)^0.75
        where f(w) is the frequency of word w
        
        Returns:
            Normalized probability array for all word IDs
        """
        probs = np.zeros(self.vocab.vocab_size)
        
        for word_id in range(self.vocab.vocab_size):
            word = self.vocab.get_word(word_id)
            word_freq = self.vocab.word_freq.get(word, 1)  # Avoid zero frequency
            
            # Apply 3/4 power (0.75 exponent)
            probs[word_id] = word_freq ** 0.75
        
        # Normalize to create probability distribution
        probs = probs / probs.sum()
        return probs
    
    def sample_negatives(self, positive_word_id: int) -> List[int]:
        """
        Sample negative words for a given positive word.
        Ensures negatives don't include the positive word.
        
        Args:
            positive_word_id: Word ID of the positive example
            
        Returns:
            List of negative word IDs
        """
        negatives = []
        attempts = 0
        max_attempts = self.num_negatives * 20  # Prevent infinite loops
        
        while len(negatives) < self.num_negatives and attempts < max_attempts:
            # Sample according to 3/4 power distribution
            neg_word_id = np.random.choice(self.word_ids, p=self.sampling_probs)
            
            # Ensure it's not the positive word and not already selected
            if neg_word_id != positive_word_id and neg_word_id not in negatives:
                negatives.append(neg_word_id)
            
            attempts += 1
        
        # Fill remaining slots if needed (fallback)
        while len(negatives) < self.num_negatives:
            neg_word_id = random.randint(0, self.vocab.vocab_size - 1)
            if neg_word_id != positive_word_id and neg_word_id not in negatives:
                negatives.append(neg_word_id)
        
        return negatives
    
    def sample_multiple_negatives(self, positive_word_ids: List[int]) -> List[List[int]]:
        """
        Sample negatives for multiple positive words.
        
        Args:
            positive_word_ids: List of positive word IDs
            
        Returns:
            List of negative word ID lists (one per positive word)
        """
        return [self.sample_negatives(pos_id) for pos_id in positive_word_ids]


# =============================================================================
# UTILITY FUNCTIONS FOR NEGATIVE SAMPLING ANALYSIS
# =============================================================================

def print_sampling_distribution(neg_sampler: SimpleNegativeSampler, top_n: int = 20) -> None:
    """
    Display the negative sampling probability distribution.
    
    Args:
        neg_sampler: SimpleNegativeSampler instance
        top_n: Number of top words to display
    """
    print(f"\n{'='*60}")
    print(f"NEGATIVE SAMPLING DISTRIBUTION")
    print(f"{'='*60}")
    print(f"Formula: P(w) ∝ f(w)^0.75 (3/4 power of frequency)")
    print(f"Total vocabulary size: {neg_sampler.vocab.vocab_size}")
    print("-" * 60)
    
    # Create list of (word, original_freq, sampling_prob)
    word_data = []
    for word_id in range(neg_sampler.vocab.vocab_size):
        word = neg_sampler.vocab.get_word(word_id)
        if word not in neg_sampler.vocab.special_tokens:
            original_freq = neg_sampler.vocab.word_freq[word]
            sampling_prob = neg_sampler.sampling_probs[word_id]
            word_data.append((word, original_freq, sampling_prob, word_id))
    
    # Sort by sampling probability (highest first)
    word_data.sort(key=lambda x: x[2], reverse=True)
    
    print(f"{'Word':<15} {'Orig.Freq':<10} {'Samp.Prob':<12} {'Prob%':<8} {'Ratio':<8}")
    print("-" * 60)
    
    for word, orig_freq, samp_prob, word_id in word_data[:top_n]:
        prob_percent = samp_prob * 100
        
        # Calculate ratio of sampling prob to original frequency proportion
        orig_freq_prop = orig_freq / neg_sampler.vocab.total_words
        ratio = samp_prob / orig_freq_prop if orig_freq_prop > 0 else 0
        
        print(f"{word:<15} {orig_freq:<10,} {samp_prob:<12.6f} {prob_percent:<8.3f} {ratio:<8.2f}")
    
    # Show distribution statistics
    print(f"\nDistribution Statistics:")
    print("-" * 40)
    
    all_probs = [data[2] for data in word_data]
    print(f"Sum of probabilities: {sum(all_probs):.6f} (should be ~1.0)")
    print(f"Max probability: {max(all_probs):.6f}")
    print(f"Min probability: {min(all_probs):.6f}")
    print(f"Mean probability: {np.mean(all_probs):.6f}")
    
    # Show effect of 3/4 power
    print(f"\nEffect of 3/4 Power:")
    print("-" * 40)
    print("The 3/4 power smooths the distribution:")
    print("- Makes frequent words less dominant")
    print("- Gives rare words better sampling chances")
    print("- Balances between uniform and frequency-based sampling")
    
    print("-" * 60)


def print_negative_sampling_demo(neg_sampler: SimpleNegativeSampler, 
                               demo_words: List[str], 
                               num_runs: int = 5) -> None:
    """
    Demonstrate negative sampling for specific words.
    
    Args:
        neg_sampler: SimpleNegativeSampler instance
        demo_words: List of words to demonstrate
        num_runs: Number of sampling runs per word
    """
    print(f"\n{'='*60}")
    print(f"NEGATIVE SAMPLING DEMONSTRATION")
    print(f"{'='*60}")
    print(f"Sampling {neg_sampler.num_negatives} negatives per positive word")
    print(f"Running {num_runs} trials per word to show variability")
    print("-" * 60)
    
    for word in demo_words:
        word_id = neg_sampler.vocab.get_word_id(word)
        actual_word = neg_sampler.vocab.get_word(word_id)  # Handle UNK case
        
        print(f"\nPositive word: '{actual_word}' (ID: {word_id})")
        
        if word != actual_word:
            print(f"  Note: '{word}' not in vocabulary, using <UNK>")
        
        print(f"Negative sampling runs:")
        
        # Track frequency of sampled negatives across runs
        negative_counter = Counter()
        
        for run in range(num_runs):
            negatives = neg_sampler.sample_negatives(word_id)
            negative_words = [neg_sampler.vocab.get_word(neg_id) for neg_id in negatives]
            
            # Count occurrences
            for neg_word in negative_words:
                negative_counter[neg_word] += 1
            
            print(f"  Run {run+1}: {negative_words}")
        
        # Show frequency of sampled negatives
        if negative_counter:
            print(f"  Frequency across {num_runs} runs:")
            most_common = negative_counter.most_common(10)
            for neg_word, count in most_common:
                percentage = (count / (num_runs * neg_sampler.num_negatives)) * 100
                print(f"    {neg_word}: {count}/{num_runs * neg_sampler.num_negatives} ({percentage:.1f}%)")
        
        print("-" * 40)


def analyze_negative_sampling_statistics(neg_sampler: SimpleNegativeSampler, 
                                       num_samples: int = 1000) -> None:
    """
    Analyze statistical properties of negative sampling.
    
    Args:
        neg_sampler: SimpleNegativeSampler instance
        num_samples: Number of sampling trials for analysis
    """
    print(f"\n{'='*60}")
    print(f"NEGATIVE SAMPLING STATISTICAL ANALYSIS")
    print(f"{'='*60}")
    print(f"Analyzing {num_samples:,} negative sampling trials")
    print("-" * 60)
    
    # Sample negatives for random positive words
    sampled_negatives = []
    positive_words_used = []
    
    for _ in range(num_samples):
        # Pick a random positive word (excluding special tokens)
        regular_word_ids = [wid for wid in range(neg_sampler.vocab.vocab_size) 
                           if neg_sampler.vocab.get_word(wid) not in neg_sampler.vocab.special_tokens]
        
        if regular_word_ids:
            pos_word_id = random.choice(regular_word_ids)
            positive_words_used.append(pos_word_id)
            
            negatives = neg_sampler.sample_negatives(pos_word_id)
            sampled_negatives.extend(negatives)
    
    # Analyze the distribution of sampled negatives
    negative_counter = Counter(sampled_negatives)
    total_negatives = len(sampled_negatives)
    
    print(f"Sampling Results:")
    print(f"  - Total negative samples: {total_negatives:,}")
    print(f"  - Unique words sampled: {len(negative_counter):,}")
    print(f"  - Coverage: {len(negative_counter) / neg_sampler.vocab.vocab_size * 100:.1f}% of vocabulary")
    
    # Most frequently sampled negatives
    print(f"\nMost Frequently Sampled Negatives:")
    print("-" * 50)
    print(f"{'Word':<15} {'Count':<8} {'Actual%':<10} {'Expected%':<12} {'Ratio':<8}")
    print("-" * 50)
    
    most_common = negative_counter.most_common(15)
    for word_id, count in most_common:
        word = neg_sampler.vocab.get_word(word_id)
        actual_percent = (count / total_negatives) * 100
        expected_percent = neg_sampler.sampling_probs[word_id] * 100
        ratio = actual_percent / expected_percent if expected_percent > 0 else 0
        
        print(f"{word:<15} {count:<8,} {actual_percent:<10.3f} {expected_percent:<12.3f} {ratio:<8.2f}")
    
    # Distribution quality metrics
    print(f"\nDistribution Quality:")
    print("-" * 40)
    
    # Calculate chi-square-like statistic for goodness of fit
    chi_square = 0
    for word_id, count in negative_counter.items():
        expected = neg_sampler.sampling_probs[word_id] * total_negatives
        if expected > 0:
            chi_square += ((count - expected) ** 2) / expected
    
    print(f"Chi-square statistic: {chi_square:.2f}")
    print(f"Expected range for good fit: Lower values indicate better match to theoretical distribution")
    
    # Show rare vs common word sampling
    word_freq_ranges = {
        'Very Common (top 10%)': [],
        'Common (10-30%)': [],
        'Medium (30-70%)': [],
        'Rare (70-90%)': [],
        'Very Rare (bottom 10%)': []
    }
    
    # Sort words by original frequency
    word_freq_list = [(wid, neg_sampler.vocab.word_freq[neg_sampler.vocab.get_word(wid)]) 
                     for wid in range(neg_sampler.vocab.vocab_size)
                     if neg_sampler.vocab.get_word(wid) not in neg_sampler.vocab.special_tokens]
    word_freq_list.sort(key=lambda x: x[1], reverse=True)
    
    # Categorize words
    total_words = len(word_freq_list)
    for i, (word_id, freq) in enumerate(word_freq_list):
        percentile = i / total_words
        count = negative_counter.get(word_id, 0)
        
        if percentile < 0.1:
            word_freq_ranges['Very Common (top 10%)'].append(count)
        elif percentile < 0.3:
            word_freq_ranges['Common (10-30%)'].append(count)
        elif percentile < 0.7:
            word_freq_ranges['Medium (30-70%)'].append(count)
        elif percentile < 0.9:
            word_freq_ranges['Rare (70-90%)'].append(count)
        else:
            word_freq_ranges['Very Rare (bottom 10%)'].append(count)
    
    print(f"\nSampling by Word Frequency Range:")
    print("-" * 50)
    print(f"{'Range':<25} {'Avg Samples':<12} {'Total Samples':<15}")
    print("-" * 50)
    
    for range_name, counts in word_freq_ranges.items():
        if counts:
            avg_samples = np.mean(counts)
            total_samples = sum(counts)
            print(f"{range_name:<25} {avg_samples:<12.2f} {total_samples:<15,}")
    
    print("-" * 60)


def compare_sampling_distributions(neg_sampler: SimpleNegativeSampler) -> None:
    """
    Compare different sampling distributions (uniform, frequency-based, 3/4 power).
    
    Args:
        neg_sampler: SimpleNegativeSampler instance
    """
    print(f"\n{'='*60}")
    print(f"SAMPLING DISTRIBUTION COMPARISON")
    print(f"{'='*60}")
    print("Comparing three sampling strategies:")
    print("1. Uniform: Equal probability for all words")
    print("2. Frequency: P(w) ∝ f(w)")
    print("3. 3/4 Power: P(w) ∝ f(w)^0.75 (Word2Vec default)")
    print("-" * 60)
    
    # Get word data
    word_data = []
    for word_id in range(min(20, neg_sampler.vocab.vocab_size)):  # Top 20 for display
        word = neg_sampler.vocab.get_word(word_id)
        if word not in neg_sampler.vocab.special_tokens:
            freq = neg_sampler.vocab.word_freq[word]
            word_data.append((word, freq, word_id))
    
    # Sort by frequency (descending)
    word_data.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate different distributions
    total_freq = sum(data[1] for data in word_data)
    
    print(f"{'Word':<12} {'Frequency':<10} {'Uniform%':<10} {'Freq%':<10} {'3/4 Pow%':<12}")
    print("-" * 60)
    
    for word, freq, word_id in word_data:
        # Uniform distribution
        uniform_prob = (1 / len(word_data)) * 100
        
        # Frequency-based distribution
        freq_prob = (freq / total_freq) * 100
        
        # 3/4 power distribution
        power_prob = neg_sampler.sampling_probs[word_id] * 100
        
        print(f"{word:<12} {freq:<10,} {uniform_prob:<10.2f} {freq_prob:<10.2f} {power_prob:<12.2f}")
    
    print(f"\nKey Insights:")
    print("-" * 40)
    print("• Uniform: Treats all words equally (may oversample rare words)")
    print("• Frequency: Heavily favors common words (may undersample rare words)")
    print("• 3/4 Power: Balanced approach that gives rare words better chances")
    print("• The 3/4 power reduces the dominance of very frequent words")
    print("• This helps the model learn better representations for all words")
    
    print("-" * 60)


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Import required modules
    try:
        from vocab_builder import SimpleVocabulary
    except ImportError:
        print("This module requires SimpleVocabulary. Please ensure it's available.")
        exit(1)
    
    # Sample data for testing
    sample_sentences = [
        ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'],
        ['the', 'dog', 'is', 'very', 'lazy', 'and', 'sleeps', 'all', 'day'],
        ['word2vec', 'learns', 'word', 'embeddings', 'from', 'text', 'data'],
        ['the', 'algorithm', 'uses', 'the', 'skip', 'gram', 'model'],
        ['frequent', 'words', 'like', 'the', 'and', 'is', 'appear', 'often'],
        ['negative', 'sampling', 'improves', 'training', 'efficiency', 'significantly'],
        ['the', 'model', 'learns', 'to', 'distinguish', 'positive', 'from', 'negative'],
        ['this', 'technique', 'was', 'introduced', 'in', 'the', 'word2vec', 'paper']
    ]
    
    print("Testing SimpleNegativeSampler with sample data...")
    
    # Build vocabulary first
    vocab = SimpleVocabulary(min_freq=1, max_size=None)
    vocab.build_from_sentences(sample_sentences)
    
    # Test negative sampling with different numbers of negatives
    num_negatives = 3
    # for num_negatives in [3, 5, 10]:
    print(f"\n{'#'*80}")
    print(f"TESTING WITH {num_negatives} NEGATIVES PER POSITIVE")
    print(f"{'#'*80}")
    
    # Create negative sampler
    neg_sampler = SimpleNegativeSampler(vocab, num_negatives)
    
    # Show sampling distribution
    # print_sampling_distribution(neg_sampler, top_n=15)
    
    # Demonstrate sampling
    demo_words = ['the', 'word2vec', 'fox', 'unknown_word']
    print_negative_sampling_demo(neg_sampler, demo_words, num_runs=3)
    
    # Statistical analysis
    # analyze_negative_sampling_statistics(neg_sampler, num_samples=500)
    
    # Compare distributions
    # compare_sampling_distributions(neg_sampler)

    print(f"\n{'='*60}")
    print("SimpleNegativeSampler testing complete!")
    print(f"{'='*60}")