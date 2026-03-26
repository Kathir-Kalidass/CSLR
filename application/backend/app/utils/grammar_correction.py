"""
Grammar Correction (Production-Ready)
Rule-based gloss-to-sentence translation for Indian Sign Language
"""

from __future__ import annotations

from typing import Dict, List, Optional


class GrammarCorrector:
    """
    Rule-based gloss-to-sentence converter for ISL/ASL.
    
    Handles:
    - Pattern matching
    - Pronoun conversion
    - Word ordering
    - Punctuation insertion
    """

    def __init__(self, patterns: Optional[Dict[str, str]] = None) -> None:
        """
        Args:
            patterns: Custom gloss pattern mappings
        """
        # Default ISL/ASL patterns
        self.patterns = patterns or {
            "ME GO SCHOOL": "I am going to school.",
            "ME NEED WATER": "I need water.",
            "WHAT YOUR NAME": "What is your name?",
            "YOU FINE": "Are you fine?",
            "HELLO THANK YOU": "Hello, thank you.",
            "THANK YOU": "Thank you.",
            "PLEASE HELP": "Please help me.",
            "GOOD MORNING": "Good morning.",
            "HOW YOU": "How are you?",
            "ME HAPPY": "I am happy.",
            "ME SAD": "I am sad.",
            "ME HUNGRY": "I am hungry.",
            "ME THIRSTY": "I am thirsty.",
            "WHAT TIME": "What time is it?",
            "WHERE YOU": "Where are you?",
            "NICE MEET YOU": "Nice to meet you.",
        }
        
        # Pronoun mappings
        self.pronoun_map = {
            "ME": "I",
            "YOU": "you",
            "HE": "he",
            "SHE": "she",
            "WE": "we",
            "THEY": "they",
        }
        
        # Verb mappings (for basic conjugation)
        self.verb_map = {
            "GO": "am going",
            "COME": "am coming",
            "EAT": "am eating",
            "DRINK": "am drinking",
            "NEED": "need",
            "WANT": "want",
            "LIKE": "like",
            "LOVE": "love",
        }

    def gloss_to_sentence(self, gloss_tokens: List[str]) -> str:
        """
        Convert gloss tokens to natural language sentence.

        Args:
            gloss_tokens: List of gloss tokens (e.g., ["ME", "GO", "SCHOOL"])

        Returns:
            Natural language sentence
        """
        if not gloss_tokens:
            return ""

        text = " ".join(gloss_tokens).upper()

        # Check for exact pattern match
        if text in self.patterns:
            return self.patterns[text]

        # Apply rules for basic ISL structure: SUBJECT VERB OBJECT
        tokens = gloss_tokens.copy()
        
        # Handle pronoun + verb patterns
        if len(tokens) >= 2:
            subject = tokens[0]
            verb = tokens[1]
            
            if subject in self.pronoun_map:
                # Convert ME -> I, YOU -> you, etc.
                if subject == "ME":
                    if verb in self.verb_map:
                        # ME GO SCHOOL -> I am going to school
                        rest = " ".join(tokens[2:]).lower() if len(tokens) > 2 else ""
                        verb_conjugated = self.verb_map.get(verb, verb.lower())
                        
                        # Determine punctuation
                        if tokens[0] in ["WHAT", "WHERE", "HOW", "WHO"]:
                            punct = "?"
                        elif tokens[0] in ["HELLO", "THANK", "PLEASE"]:
                            punct = "!"
                        else:
                            punct = "."
                        
                        if rest:
                            if rest in ["school", "home", "work", "hospital"]:
                                return f"I {verb_conjugated} to {rest}{punct}"
                            else:
                                return f"I {verb_conjugated} {rest}{punct}"
                        else:
                            return f"I {verb_conjugated}{punct}"
                    else:
                        # ME + other verb
                        rest = " ".join(tokens[1:]).lower()
                        return f"I {rest}."
                
                elif subject == "YOU":
                    # YOU + rest
                    rest = " ".join(tokens[1:]).lower()
                    
                    # Check if question
                    if verb in ["WHAT", "WHERE", "HOW", "WHO"]:
                        return f"{rest.capitalize()}?"
                    else:
                        return f"You {rest}."

        # Handle question words
        if tokens[0] in ["WHAT", "WHERE", "HOW", "WHO", "WHEN", "WHY"]:
            rest = " ".join(tokens[1:]).lower()
            question_word = tokens[0].capitalize()
            return f"{question_word} {rest}?"

        # Handle greetings/thanks
        if tokens[0] in ["HELLO", "HI", "BYE", "GOODBYE"]:
            rest = " ".join(tokens[1:]).lower() if len(tokens) > 1 else ""
            greeting = tokens[0].capitalize()
            if rest:
                return f"{greeting}, {rest}!"
            return f"{greeting}!"

        if "THANK" in tokens or "THANKS" in tokens:
            return "Thank you!"

        if "PLEASE" in tokens:
            rest = " ".join([t for t in tokens if t != "PLEASE"]).lower()
            if rest:
                return f"Please {rest}."
            return "Please!"

        # Default: capitalize first word and add period
        sentence = " ".join(tokens).lower().capitalize()
        
        # Add appropriate punctuation
        if any(q in tokens for q in ["WHAT", "WHERE", "HOW", "WHO", "WHEN"]):
            return f"{sentence}?"
        elif any(e in tokens for e in ["HELLO", "HI", "THANK", "THANKS"]):
            return f"{sentence}!"
        else:
            return f"{sentence}."

    def add_pattern(self, gloss: str, sentence: str) -> None:
        """
        Add a custom pattern.

        Args:
            gloss: Gloss pattern (space-separated)
            sentence: Corresponding sentence
        """
        self.patterns[gloss.upper()] = sentence

    def batch_convert(self, gloss_sequences: List[List[str]]) -> List[str]:
        """
        Convert multiple gloss sequences.

        Args:
            gloss_sequences: List of gloss token sequences

        Returns:
            List of sentences
        """
        return [self.gloss_to_sentence(seq) for seq in gloss_sequences]
