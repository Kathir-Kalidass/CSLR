"""
Grammar Corrector
Fixes grammatical errors in translated text
"""

from typing import List
import re


class GrammarCorrector:
    """
    Applies grammar correction to translated text
    Can use rule-based or model-based correction
    """
    
    def __init__(self):
        self.article_rules = self._load_article_rules()
        self.verb_conjugations = self._load_verb_conjugations()
    
    def _load_article_rules(self) -> dict:
        """Simple article insertion rules"""
        return {
            # Words that typically need articles
            "a": ["book", "car", "house", "person", "apple"],
            "the": ["sun", "moon", "earth", "world"],
        }
    
    def _load_verb_conjugations(self) -> dict:
        """Basic verb conjugation rules"""
        return {
            "be": {
                "I": "am",
                "you": "are",
                "he": "is",
                "she": "is",
                "it": "is",
                "we": "are",
                "they": "are"
            },
            "have": {
                "I": "have",
                "you": "have",
                "he": "has",
                "she": "has",
                "it": "has",
                "we": "have",
                "they": "have"
            }
        }
    
    def correct(self, text: str) -> str:
        """
        Apply grammar corrections
        
        Args:
            text: Input text
        
        Returns:
            Corrected text
        """
        if not text:
            return text
        
        # Apply various correction rules
        text = self._fix_capitalization(text)
        text = self._fix_spacing(text)
        text = self._add_articles(text)
        text = self._fix_verb_forms(text)
        
        return text

    def gloss_to_sentence(self, gloss_tokens: List[str]) -> str:
        """
        Backward-compatible API: convert gloss tokens to a sentence string.
        """
        if not gloss_tokens:
            return ""

        pronoun_map = {
            "IX-1p": "I",
            "IX-2p": "you",
            "IX-3p": "he",
        }
        words = [pronoun_map.get(token, token.lower()) for token in gloss_tokens]
        text = " ".join(words)
        return self.correct(text)
    
    def _fix_capitalization(self, text: str) -> str:
        """Fix capitalization"""
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        # Capitalize I
        text = re.sub(r'\bi\b', 'I', text)
        
        return text
    
    def _fix_spacing(self, text: str) -> str:
        """Fix spacing issues"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?])', r'\1', text)
        
        return text.strip()
    
    def _add_articles(self, text: str) -> str:
        """Add missing articles (basic)"""
        # TODO: Implement smarter article insertion
        return text
    
    def _fix_verb_forms(self, text: str) -> str:
        """Fix verb conjugations (basic)"""
        # TODO: Implement verb form correction
        return text
    
    def correct_sequence(self, words: List[str]) -> List[str]:
        """
        Correct word sequence before joining
        
        Args:
            words: List of words
        
        Returns:
            Corrected word list
        """
        if not words:
            return words
        
        # Capitalize first word
        words[0] = words[0].capitalize()
        
        # Capitalize "I"
        words = [w if w.lower() != "i" else "I" for w in words]
        
        return words
