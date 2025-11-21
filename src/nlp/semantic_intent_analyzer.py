#
# Semantic Intent Analyzer - نظام تحليل النوايا الدلالية
# =========================================================
#
# Analyzes text to determine semantic meaning, user intent, and emotional tone.
# Part of the Deep Cognition v1.2 Lite system.
#
# Author: Noogh AI Team
# Version: 1.1.0 (Upgraded from stub to rule-based implementation)
#

from dataclasses import dataclass, field
from enum import Enum
import re

# Define structured enums for different layers of analysis
class SemanticLayer(str, Enum):
    """The semantic context of the text."""
    TECHNICAL = "technical"      # Code, files, systems
    GENERAL = "general"          # General questions, facts
    PHILOSOPHICAL = "philosophical" # Abstract ideas, opinions
    CONVERSATIONAL = "conversational" # Greetings, chit-chat

class IntentLayer(str, Enum):
    """The user's intent."""
    QUERY = "query"              # Asking for information (what, how, why)
    COMMAND = "command"          # Telling the system to do something (run, create, analyze)
    COMPARISON = "comparison"    # Comparing two or more things (vs, compare)
    CONFIRMATION = "confirmation"  # Confirming something (yes, ok, proceed)
    NEGATION = "negation"        # Denying or stopping something (no, stop, cancel)

class EmotionalTone(str, Enum):
    """The emotional tone of the text."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    INQUISITIVE = "inquisitive" # Questioning

# Dataclasses for structuring the analysis output
@dataclass
class Emotional:
    tone: EmotionalTone
    intensity: float = 0.0

@dataclass
class Analysis:
    semantic: SemanticLayer
    intent: IntentLayer
    emotional: Emotional
    semantic_intent_alignment: float = 0.0
    interpreted_meaning: str = ""
    contradiction_detected: bool = False
    contradiction_description: str = ""

class SemanticIntentAnalyzer:
    """
    Analyzes text using rule-based methods to determine semantic meaning,
    user intent, and emotional tone.
    """
    def __init__(self):
        # Define keywords for different layers
        self.intent_keywords = {
            IntentLayer.QUERY: ['what', 'how', 'why', 'who', 'when', 'where', 'explain', 'describe', 'ما', 'كيف', 'لماذا', 'اشرح'],
            IntentLayer.COMMAND: ['run', 'execute', 'create', 'delete', 'write', 'read', 'analyze', 'start', 'stop', 'شغل', 'نفذ', 'أنشئ', 'احذف', 'اقرأ', 'حلل'],
            IntentLayer.COMPARISON: ['vs', 'versus', 'compare', 'difference', 'مقارنة', 'الفرق'],
            IntentLayer.CONFIRMATION: ['yes', 'ok', 'okay', 'proceed', 'continue', 'good', 'great', 'نعم', 'حسنا', 'موافق', 'اكمل'],
            IntentLayer.NEGATION: ['no', 'not', 'stop', 'cancel', 'don\'t', 'لا', 'توقف', 'الغاء'],
        }
        self.semantic_keywords = {
            SemanticLayer.TECHNICAL: ['code', 'file', 'system', 'error', 'bug', 'python', 'javascript', 'docker', 'api', 'كود', 'ملف', 'نظام', 'خطأ'],
            SemanticLayer.PHILOSOPHICAL: ['think', 'believe', 'opinion', 'why', 'meaning', 'purpose', 'اعتقد', 'رأيك', 'لماذا', 'معنى'],
        }
        self.emotional_keywords = {
            EmotionalTone.POSITIVE: ['good', 'great', 'excellent', 'love', 'thanks', 'thank you', 'awesome', 'جيد', 'ممتاز', 'رائع', 'شكرا'],
            EmotionalTone.NEGATIVE: ['bad', 'error', 'problem', 'hate', 'issue', 'fail', 'سيء', 'مشكلة', 'خطأ', 'فشل'],
        }

    def analyze(self, text: str) -> Analysis:
        """
        Performs a rule-based analysis of the input text.
        
        Args:
            text: The user input string.
            
        Returns:
            An Analysis object with the results.
        """
        if not isinstance(text, str) or not text:
            return Analysis(
                semantic=SemanticLayer.CONVERSATIONAL,
                intent=IntentLayer.QUERY,
                emotional=Emotional(tone=EmotionalTone.NEUTRAL, intensity=0.0),
                interpreted_meaning="No input provided."
            )

        text_lower = text.lower()
        words = set(re.findall(r'\w+', text_lower))

        # 1. Determine Intent
        intent = self._get_layer(words, self.intent_keywords, IntentLayer.QUERY)

        # 2. Determine Semantic Layer
        semantic = self._get_layer(words, self.semantic_keywords, SemanticLayer.GENERAL)
        if len(words) <= 3:
            semantic = SemanticLayer.CONVERSATIONAL

        # 3. Determine Emotional Tone
        positive_score = len([w for w in self.emotional_keywords[EmotionalTone.POSITIVE] if w in words])
        negative_score = len([w for w in self.emotional_keywords[EmotionalTone.NEGATIVE] if w in words])
        
        if positive_score > negative_score:
            tone = EmotionalTone.POSITIVE
            intensity = min(1.0, positive_score * 0.3)
        elif negative_score > positive_score:
            tone = EmotionalTone.NEGATIVE
            intensity = min(1.0, negative_score * 0.3)
        else:
            tone = EmotionalTone.INQUISITIVE if '?' in text or intent == IntentLayer.QUERY else EmotionalTone.NEUTRAL
            intensity = 0.1 if tone == EmotionalTone.INQUISITIVE else 0.0
        
        emotional_analysis = Emotional(tone=tone, intensity=intensity)

        # 4. Calculate Alignment (simple version)
        # High alignment if intent and semantics match (e.g., technical words with a command)
        alignment = 0.5 # Base alignment
        if semantic == SemanticLayer.TECHNICAL and intent == IntentLayer.COMMAND:
            alignment = 0.9
        elif semantic == SemanticLayer.GENERAL and intent == IntentLayer.QUERY:
            alignment = 0.8
        
        # 5. Detect Contradiction (Simple rule: Positive tone with negative words or vice versa)
        contradiction_detected = False
        contradiction_description = ""
        if tone == EmotionalTone.POSITIVE and negative_score > 0:
             contradiction_detected = True
             contradiction_description = "Positive tone detected but negative words present (Sarcasm?)"
        elif tone == EmotionalTone.NEGATIVE and positive_score > 0:
             contradiction_detected = True
             contradiction_description = "Negative tone detected but positive words present (Mixed signals?)"

        # 6. Generate Interpreted Meaning
        meaning = f"User issued a '{intent.value}' with a '{semantic.value}' context and a '{tone.value}' tone."

        return Analysis(
            semantic=semantic,
            intent=intent,
            emotional=emotional_analysis,
            semantic_intent_alignment=alignment,
            interpreted_meaning=meaning,
            contradiction_detected=contradiction_detected,
            contradiction_description=contradiction_description
        )

    def analyze_contextual(self, text: str, history: list = None) -> Analysis:
        """
        Analyzes text considering the conversation history.
        
        Args:
            text: Current user input
            history: List of previous Analysis objects or dicts
            
        Returns:
            Context-aware Analysis object
        """
        # 1. Basic Analysis
        current_analysis = self.analyze(text)
        
        if not history:
            return current_analysis
            
        # 2. Contextual Refinement
        last_turn = history[-1] if history else None
        
        # Handle follow-up questions (e.g., "why?", "and then?")
        # Relaxed condition: length <= 7 or starts with connector
        words = text.lower().split()
        is_short = len(words) <= 7
        connectors = ["and", "but", "so", "why", "how", "what", "then"]
        starts_with_connector = words[0] in connectors if words else False
        
        if (is_short or starts_with_connector) and last_turn:
            # If previous was a query, this is likely a follow-up query
            if isinstance(last_turn, dict):
                last_intent = last_turn.get('intent')
                last_semantic = last_turn.get('semantic')
            else:
                last_intent = last_turn.intent
                last_semantic = last_turn.semantic
                
            if last_intent == IntentLayer.QUERY:
                current_analysis.intent = IntentLayer.QUERY
                current_analysis.semantic = last_semantic
                current_analysis.interpreted_meaning += " (Follow-up)"
                
        return current_analysis

    def _get_layer(self, words: set, keyword_map: dict, default_layer: Enum):
        """Helper to find the best matching layer based on keywords."""
        scores = {layer: 0 for layer in keyword_map}
        for layer, keywords in keyword_map.items():
            scores[layer] = len([w for w in keywords if w in words])
        
        # Find the layer with the highest score
        if any(s > 0 for s in scores.values()):
            best_layer = max(scores, key=scores.get)
            return best_layer
        
        return default_layer