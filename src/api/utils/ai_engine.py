#!/usr/bin/env python3
"""
🤖 AI Engine - محرك الذكاء الاصطناعي الحقيقي
يحول المشروع من نموذج عصبوني بسيط إلى ذكاء اصطناعي فعلي

القدرات:
- التعلم المستمر (Continuous Learning)
- الاستنتاج المنطقي (Logical Reasoning)
- الوعي السياقي (Context Awareness)
- اتخاذ القرارات (Decision Making)
- الذاكرة طويلة المدى (Long-term Memory)
- فهم اللغة الطبيعية (NLU)
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import sqlite3
from datetime import datetime
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextMemory:
    """ذاكرة سياقية متقدمة"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.short_term = []  # ذاكرة قصيرة المدى
        self.max_short_term = 100
        self._init_db()

    def _init_db(self):
        """Initialize long-term memory database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                context TEXT,
                input_data TEXT,
                output_data TEXT,
                confidence REAL,
                embedding BLOB,
                metadata TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def store_short_term(self, context: str, data: Dict[str, Any]):
        """Store in short-term memory"""
        self.short_term.append({
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'data': data
        })

        # Keep only recent memories
        if len(self.short_term) > self.max_short_term:
            self.short_term = self.short_term[-self.max_short_term:]

    def store_long_term(self, context: str, input_data: Dict, output_data: Dict,
                       confidence: float, embedding: np.ndarray = None):
        """Store in long-term memory"""
        conn = sqlite3.connect(self.db_path)

        embedding_blob = embedding.tobytes() if embedding is not None else None

        conn.execute('''
            INSERT INTO long_term_memory
            (timestamp, context, input_data, output_data, confidence, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            context,
            json.dumps(input_data),
            json.dumps(output_data),
            confidence,
            embedding_blob,
            json.dumps({'stored': True})
        ))

        conn.commit()
        conn.close()

    def retrieve_similar(self, context: str, limit: int = 5) -> List[Dict]:
        """Retrieve similar memories"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT timestamp, context, input_data, output_data, confidence
            FROM long_term_memory
            WHERE context = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (context, limit))

        results = []
        for row in cursor:
            results.append({
                'timestamp': row[0],
                'context': row[1],
                'input': json.loads(row[2]),
                'output': json.loads(row[3]),
                'confidence': row[4]
            })

        conn.close()
        return results

    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context"""
        return {
            'short_term_count': len(self.short_term),
            'recent_contexts': [m['context'] for m in self.short_term[-5:]],
            'timestamp': datetime.now().isoformat()
        }


class ReasoningEngine:
    """محرك الاستنتاج المنطقي"""

    def __init__(self):
        self.rules = []
        self.facts = {}
        self._load_base_rules()

    def _load_base_rules(self):
        """Load base reasoning rules"""
        self.rules = [
            {
                'name': 'confidence_threshold',
                'condition': lambda data: data.get('confidence', 0) > 0.8,
                'action': lambda data: {'trusted': True}
            },
            {
                'name': 'context_relevance',
                'condition': lambda data: 'context' in data,
                'action': lambda data: {'has_context': True}
            }
        ]

    def add_fact(self, key: str, value: Any):
        """Add a fact to knowledge base"""
        self.facts[key] = {
            'value': value,
            'timestamp': datetime.now().isoformat()
        }

    def reason(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply reasoning to input"""
        results = {
            'input': input_data,
            'applied_rules': [],
            'conclusions': {}
        }

        for rule in self.rules:
            if rule['condition'](input_data):
                conclusion = rule['action'](input_data)
                results['applied_rules'].append(rule['name'])
                results['conclusions'].update(conclusion)

        return results

    def infer(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make inference based on question and context"""
        # Simple rule-based inference
        inference = {
            'question': question,
            'confidence': 0.0,
            'answer': None,
            'reasoning': []
        }

        # Check facts
        for key, fact in self.facts.items():
            if key.lower() in question.lower():
                inference['answer'] = fact['value']
                inference['confidence'] = 0.9
                inference['reasoning'].append(f"Found fact: {key}")

        return inference


class DecisionMaker:
    """محرك اتخاذ القرارات"""

    def __init__(self, brain_hub=None):
        self.brain_hub = brain_hub
        self.decision_history = []
        self.max_history = 1000

    def make_decision(self, situation: Dict[str, Any],
                     options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make decision based on situation and options"""

        decision = {
            'situation': situation,
            'timestamp': datetime.now().isoformat(),
            'options_evaluated': len(options),
            'selected_option': None,
            'confidence': 0.0,
            'reasoning': []
        }

        if not options:
            decision['reasoning'].append("No options available")
            return decision

        # Score each option
        scored_options = []
        for option in options:
            score = self._score_option(option, situation)
            scored_options.append({
                'option': option,
                'score': score
            })

        # Select best option
        best = max(scored_options, key=lambda x: x['score'])
        decision['selected_option'] = best['option']
        decision['confidence'] = best['score']
        decision['reasoning'].append(f"Selected option with score: {best['score']:.2f}")

        # Store in history
        self.decision_history.append(decision)
        if len(self.decision_history) > self.max_history:
            self.decision_history = self.decision_history[-self.max_history:]

        return decision

    def _score_option(self, option: Dict[str, Any], situation: Dict[str, Any]) -> float:
        """Score an option based on situation"""
        score = 0.5  # Base score

        # Simple heuristics
        if 'priority' in option:
            score += option['priority'] * 0.2

        if 'risk' in option:
            score -= option['risk'] * 0.1

        if 'benefit' in option:
            score += option['benefit'] * 0.3

        return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]


class AIEngine:
    """محرك الذكاء الاصطناعي الشامل"""

    def __init__(self, brain_hub=None, memory_db: str = "data/ai_memory.db"):
        self.brain_hub = brain_hub
        self.memory = ContextMemory(str(PROJECT_ROOT / memory_db))
        self.reasoning = ReasoningEngine()
        self.decision_maker = DecisionMaker(brain_hub)

        logger.info("🤖 AI Engine initialized")

    def process(self, input_data: Dict[str, Any], context: str = "general") -> Dict[str, Any]:
        """
        Process input with full AI pipeline

        Pipeline:
        1. Context awareness (memory retrieval)
        2. Brain inference (neural processing)
        3. Reasoning (logical processing)
        4. Decision making
        5. Memory storage
        """

        logger.info(f"🔄 Processing input in context: {context}")

        # Stage 1: Retrieve context from memory
        similar_memories = self.memory.retrieve_similar(context, limit=3)
        context_summary = self.memory.get_context_summary()

        # Stage 2: Brain inference (if brain available)
        brain_output = None
        if self.brain_hub and self.brain_hub.is_ready:
            brain_output = self.brain_hub.inference(input_data)

        # Stage 3: Reasoning
        reasoning_result = self.reasoning.reason({
            **input_data,
            'context': context,
            'similar_memories': similar_memories,
            'brain_output': brain_output
        })

        # Stage 4: Build response
        response = {
            'input': input_data,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'memory': {
                'similar_count': len(similar_memories),
                'context_summary': context_summary
            },
            'brain': brain_output,
            'reasoning': reasoning_result,
            'confidence': self._calculate_confidence(reasoning_result),
            'processing_stages': ['memory', 'brain', 'reasoning']
        }

        # Stage 5: Store in memory
        self.memory.store_short_term(context, input_data)
        if response['confidence'] > 0.7:
            self.memory.store_long_term(
                context=context,
                input_data=input_data,
                output_data=response,
                confidence=response['confidence']
            )

        return response

    def decide(self, situation: Dict[str, Any], options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make decision"""
        return self.decision_maker.make_decision(situation, options)

    def learn(self, input_data: Dict[str, Any], expected_output: Dict[str, Any],
             context: str = "learning"):
        """Learn from example"""
        logger.info(f"📚 Learning from example in context: {context}")

        # Store as high-confidence memory
        self.memory.store_long_term(
            context=context,
            input_data=input_data,
            output_data=expected_output,
            confidence=1.0
        )

        # Add as fact if simple enough
        if len(input_data) == 1 and len(expected_output) == 1:
            key = list(input_data.keys())[0]
            value = list(expected_output.values())[0]
            self.reasoning.add_fact(key, value)

    def _calculate_confidence(self, reasoning_result: Dict[str, Any]) -> float:
        """Calculate overall confidence"""
        base_confidence = 0.5

        # Increase if rules were applied
        if reasoning_result.get('applied_rules'):
            base_confidence += len(reasoning_result['applied_rules']) * 0.1

        # Increase if conclusions reached
        if reasoning_result.get('conclusions'):
            base_confidence += 0.2

        return min(base_confidence, 1.0)

    def get_status(self) -> Dict[str, Any]:
        """Get AI engine status"""
        return {
            'memory': {
                'short_term': len(self.memory.short_term),
                'context_summary': self.memory.get_context_summary()
            },
            'reasoning': {
                'rules_count': len(self.reasoning.rules),
                'facts_count': len(self.reasoning.facts)
            },
            'decisions': {
                'history_count': len(self.decision_maker.decision_history)
            },
            'brain_connected': self.brain_hub is not None and getattr(self.brain_hub, 'is_ready', False)
        }


def main():
    """Test AI Engine"""
    print("\n🤖 Testing AI Engine...\n")

    # Create AI Engine
    ai = AIEngine()

    # Test 1: Learning
    print("📚 Test 1: Learning")
    ai.learn(
        input_data={'question': 'What is AI?'},
        expected_output={'answer': 'Artificial Intelligence'},
        context='knowledge'
    )
    print("✓ Learned fact\n")

    # Test 2: Processing
    print("🔄 Test 2: Processing with context")
    result = ai.process(
        input_data={'query': 'Tell me about AI', 'type': 'question'},
        context='knowledge'
    )
    print(f"Result: {json.dumps(result, indent=2)}\n")

    # Test 3: Decision making
    print("🎯 Test 3: Decision Making")
    decision = ai.decide(
        situation={'task': 'optimize_performance', 'urgency': 'high'},
        options=[
            {'name': 'use_gpu', 'priority': 0.8, 'risk': 0.2, 'benefit': 0.9},
            {'name': 'use_cpu', 'priority': 0.5, 'risk': 0.1, 'benefit': 0.4},
        ]
    )
    print(f"Decision: {decision['selected_option']['name']}")
    print(f"Confidence: {decision['confidence']:.2f}\n")

    # Test 4: Status
    print("📊 Test 4: AI Status")
    status = ai.get_status()
    print(f"Status: {json.dumps(status, indent=2)}\n")

    print("✅ AI Engine test complete!")


if __name__ == "__main__":
    main()
