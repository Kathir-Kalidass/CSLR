"""
Post-Processor
Final text processing and formatting
"""

import re
from typing import List


class PostProcessor:
    """
    Final post-processing of translated text
    Handles punctuation, formatting, etc.
    """
    
    def __init__(self):
        self.punctuation_rules = self._load_punctuation_rules()
    
    def _load_punctuation_rules(self) -> dict:
        """Load punctuation insertion rules"""
        return {
            "question_words": ["what", "when", "where", "who", "why", "how"],
            "greeting_words": ["hello", "hi", "hey", "goodbye", "bye"]
        }
    
    def process(self, text: str) -> str:
        """
        Apply all post-processing steps
        
        Args:
            text: Input text
        
        Returns:
            Processed text
        """
        if not text:
            return text
        
        text = self._add_punctuation(text)
        text = self._fix_formatting(text)
        text = self._remove_repetitions(text)
        
        return text

    def apply(self, text: str) -> str:
        """
        Backward-compatible alias for process().
        """
        return self.process(text)
    
    def _add_punctuation(self, text: str) -> str:
        """Add appropriate punctuation"""
        # Check if already has ending punctuation
        if text and text[-1] in '.!?':
            return text
        
        # Check if question
        first_word = text.split()[0].lower() if text.split() else ""
        if first_word in self.punctuation_rules["question_words"]:
            text = text + '?'
        
        # Check if greeting
        elif first_word in self.punctuation_rules["greeting_words"]:
            text = text + '!'
        
        else:
            text = text + '.'
        
        return text
    
    def _fix_formatting(self, text: str) -> str:
        """Fix text formatting"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure space after punctuation
        text = re.sub(r'([,.!?])([^\s])', r'\1 \2', text)
        
        # Remove space before punctuation
        text = re.sub(r'\s+([,.!?])', r'\1', text)
        
        return text.strip()
    
    def _remove_repetitions(self, text: str) -> str:
        """Remove repeated words"""
        words = text.split()
        
        if len(words) <= 1:
            return text
        
        # Remove consecutive duplicates
        filtered = [words[0]]
        for word in words[1:]:
            if word.lower() != filtered[-1].lower():
                filtered.append(word)
        
        return ' '.join(filtered)
    
    def process_sequence(self, glosses: List[str]) -> str:
        """
        Process gloss sequence into formatted text
        
        Args:
            glosses: List of gloss tokens
        
        Returns:
            Formatted text
        """
        if not glosses:
            return ""
        
        # Convert to lowercase and join
        text = ' '.join(gloss.lower() for gloss in glosses)
        
        # Apply post-processing
        text = self.process(text)
        
        return text
    
    def clean_gloss_sequence(self, glosses: List[str]) -> List[str]:
        """
        Clean gloss sequence
        Remove duplicates, blanks, etc.
        """
        cleaned = []
        prev = None
        
        for gloss in glosses:
            # Skip blanks and special tokens
            if gloss in ["<blank>", "<pad>", "<unk>"]:
                continue
            
            # Skip consecutive duplicates
            if gloss != prev:
                cleaned.append(gloss)
            
            prev = gloss
        
        return cleaned
