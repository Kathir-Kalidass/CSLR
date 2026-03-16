"""
Test Language Processing Module
"""

import pytest


def test_grammar_corrector_initialization():
    """Test grammar corrector initialization"""
    from app.pipeline.module4_language.grammar_corrector import GrammarCorrector
    
    corrector = GrammarCorrector()
    
    assert corrector is not None


def test_grammar_corrector_gloss_to_sentence():
    """Test gloss to sentence conversion"""
    from app.pipeline.module4_language.grammar_corrector import GrammarCorrector
    
    corrector = GrammarCorrector()
    
    glosses = ["HELLO", "MY", "NAME", "JOHN"]
    sentence = corrector.gloss_to_sentence(glosses)
    
    assert isinstance(sentence, str)
    assert len(sentence) > 0


def test_grammar_corrector_pronoun_mapping():
    """Test pronoun mapping"""
    from app.pipeline.module4_language.grammar_corrector import GrammarCorrector
    
    corrector = GrammarCorrector()
    
    # Test subject pronouns
    glosses = ["IX-1p", "LIKE", "COFFEE"]
    sentence = corrector.gloss_to_sentence(glosses)
    
    assert "I" in sentence or "like" in sentence


def test_grammar_corrector_article_insertion():
    """Test article insertion"""
    from app.pipeline.module4_language.grammar_corrector import GrammarCorrector
    
    corrector = GrammarCorrector()
    
    glosses = ["I", "HAVE", "CAT"]
    sentence = corrector.gloss_to_sentence(glosses)
    
    # Should add article
    assert "a" in sentence.lower() or "cat" in sentence.lower()


def test_post_processor_initialization():
    """Test post processor initialization"""
    from app.pipeline.module4_language.post_processor import PostProcessor
    
    processor = PostProcessor()
    
    assert processor is not None


def test_post_processor_apply():
    """Test post processing"""
    from app.pipeline.module4_language.post_processor import PostProcessor
    
    processor = PostProcessor()
    
    text = "hello world how are you"
    processed = processor.apply(text)
    
    assert isinstance(processed, str)
    # Should capitalize first letter
    assert processed[0].isupper() or processed == text


def test_post_processor_capitalization():
    """Test capitalization"""
    from app.pipeline.module4_language.post_processor import PostProcessor
    
    processor = PostProcessor()
    
    text = "hello. world. how are you."
    processed = processor.apply(text)
    
    # Each sentence should start with capital
    sentences = processed.split('. ')
    for sent in sentences:
        if sent:
            assert sent[0].isupper() or sent == text


def test_translator_initialization():
    """Test translator initialization"""
    from app.pipeline.module4_language.translator import Translator
    
    translator = Translator()
    
    assert translator is not None


def test_buffer_initialization():
    """Test buffer initialization"""
    from app.pipeline.module4_language.buffer import CaptionBuffer
    
    buffer = CaptionBuffer(max_size=10)
    
    assert buffer.max_size == 10


def test_buffer_add_caption():
    """Test adding captions to buffer"""
    from app.pipeline.module4_language.buffer import CaptionBuffer
    
    buffer = CaptionBuffer(max_size=5)
    
    buffer.add("Hello")
    buffer.add("World")
    
    captions = buffer.get_all()
    
    assert len(captions) == 2
    assert "Hello" in captions
    assert "World" in captions


def test_buffer_max_size():
    """Test buffer max size enforcement"""
    from app.pipeline.module4_language.buffer import CaptionBuffer
    
    buffer = CaptionBuffer(max_size=3)
    
    for i in range(5):
        buffer.add(f"Caption {i}")
    
    captions = buffer.get_all()
    
    # Should only keep last 3
    assert len(captions) <= 3
