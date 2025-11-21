#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 Auto Data Collector - مجمع البيانات التلقائي
Automatically collects training data from various sources
"""

import asyncio
import logging
import json
import random
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class AutoDataCollector:
    """
    Automatic data collection system for training autonomous agents.

    Collects data from:
    - Knowledge base (if available)
    - Past interactions (conversation logs)
    - Synthetic data generation
    - File system scans
    """

    def __init__(self, work_dir: str = "."):
        self.work_dir = Path(work_dir)
        self.data_dir = self.work_dir / "data" / "training"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.collected_samples = 0
        self.collection_history = []

        logger.info(f"📊 AutoDataCollector initialized: {self.data_dir}")

    async def collect_training_data(
        self,
        target_samples: int,
        task_type: str = "general",
        sources: Optional[List[str]] = None
    ) -> Dict[str, List[Dict]]:
        """
        Collect training data asynchronously from multiple sources.

        Args:
            target_samples: Number of samples to collect
            task_type: Type of task (general, qa, classification, etc.)
            sources: List of data sources to use

        Returns:
            Dict with 'train' and 'test' keys containing sample lists
        """
        logger.info(f"🔄 Starting data collection: {target_samples} samples, type={task_type}")
        start_time = datetime.now()

        if sources is None:
            sources = ["knowledge_base", "synthetic", "filesystem"]

        all_samples = []

        # Collect from each source
        for source in sources:
            try:
                if source == "knowledge_base":
                    samples = await self._collect_from_knowledge_base(target_samples // len(sources))
                elif source == "synthetic":
                    samples = await self._generate_synthetic_data(target_samples // len(sources), task_type)
                elif source == "filesystem":
                    samples = await self._collect_from_filesystem(target_samples // len(sources))
                elif source == "conversations":
                    samples = await self._collect_from_conversations(target_samples // len(sources))
                else:
                    logger.warning(f"⚠️ Unknown source: {source}")
                    samples = []

                all_samples.extend(samples)
                logger.info(f"   ✅ Collected {len(samples)} samples from {source}")

            except Exception as e:
                logger.error(f"   ❌ Error collecting from {source}: {e}")

        # Shuffle and split into train/test
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * 0.8)  # 80/20 split

        train_data = all_samples[:split_idx]
        test_data = all_samples[split_idx:]

        # Ensure we have at least some data
        if len(train_data) == 0:
            logger.warning("⚠️ No training data collected, generating fallback data")
            train_data = await self._generate_synthetic_data(target_samples, task_type)
            test_data = await self._generate_synthetic_data(target_samples // 4, task_type)

        duration = (datetime.now() - start_time).total_seconds()
        self.collected_samples += len(all_samples)

        result = {
            "train": train_data,
            "test": test_data,
            "metadata": {
                "total_samples": len(all_samples),
                "train_count": len(train_data),
                "test_count": len(test_data),
                "task_type": task_type,
                "sources": sources,
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat()
            }
        }

        logger.info(f"✅ Data collection complete: {len(train_data)} train, {len(test_data)} test ({duration:.1f}s)")

        # Save collection record
        self._save_collection_record(result["metadata"])

        return result

    async def _collect_from_knowledge_base(self, num_samples: int) -> List[Dict]:
        """Collect samples from knowledge base files"""
        samples = []

        # Look for knowledge index
        kb_paths = [
            self.work_dir / "data" / "knowledge_index.json",
            self.work_dir / "knowledge_index.json",
        ]

        for kb_path in kb_paths:
            if kb_path.exists():
                try:
                    with open(kb_path, 'r', encoding='utf-8') as f:
                        kb_data = json.load(f)

                    # Sample from knowledge base
                    available = kb_data if isinstance(kb_data, list) else []
                    sampled = random.sample(available, min(num_samples, len(available)))

                    for item in sampled:
                        samples.append({
                            "text": item.get("chunk", item.get("full_chunk", str(item))),
                            "source": "knowledge_base",
                            "metadata": item.get("metadata", {})
                        })

                    logger.info(f"   📚 Found knowledge base with {len(available)} items")
                    break

                except Exception as e:
                    logger.warning(f"   ⚠️ Could not load knowledge base from {kb_path}: {e}")

        return samples[:num_samples]

    async def _generate_synthetic_data(self, num_samples: int, task_type: str) -> List[Dict]:
        """Generate synthetic training data"""
        samples = []

        # Task-specific templates
        templates = {
            "general": [
                "What is {topic}?",
                "Explain {topic} in simple terms",
                "Tell me about {topic}",
                "How does {topic} work?",
            ],
            "qa": [
                "Q: {question} A: {answer}",
                "Question: {question} Answer: {answer}",
            ],
            "classification": [
                "Category: {category} Text: {text}",
            ]
        }

        topics = ["AI", "machine learning", "Python", "data science", "neural networks",
                 "deep learning", "NLP", "computer vision", "algorithms", "optimization"]

        template_list = templates.get(task_type, templates["general"])

        for i in range(num_samples):
            topic = random.choice(topics)
            template = random.choice(template_list)

            text = template.format(
                topic=topic,
                question=f"What is {topic}?",
                answer=f"{topic} is an important concept in AI",
                category=topic,
                text=f"This text is about {topic}"
            )

            samples.append({
                "text": text,
                "source": "synthetic",
                "task_type": task_type,
                "label": i % 5  # Dummy label
            })

        # Simulate async work
        await asyncio.sleep(0.1)

        return samples

    async def _collect_from_filesystem(self, num_samples: int) -> List[Dict]:
        """Collect samples from text files in project"""
        samples = []

        # Look for .txt, .md files
        text_files = []
        for ext in [".txt", ".md"]:
            text_files.extend(list(self.work_dir.rglob(f"*{ext}")))

        # Sample files
        sampled_files = random.sample(text_files, min(num_samples, len(text_files)))

        for file_path in sampled_files:
            try:
                if file_path.stat().st_size > 1000000:  # Skip files > 1MB
                    continue

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Split into chunks
                chunks = [content[i:i+500] for i in range(0, len(content), 500)]
                chunk = random.choice(chunks) if chunks else content

                samples.append({
                    "text": chunk.strip(),
                    "source": "filesystem",
                    "file": str(file_path.relative_to(self.work_dir))
                })

            except Exception as e:
                logger.debug(f"Could not read {file_path}: {e}")

        await asyncio.sleep(0.1)
        return samples[:num_samples]

    async def _collect_from_conversations(self, num_samples: int) -> List[Dict]:
        """Collect samples from conversation logs"""
        samples = []

        # Look for conversation/log files
        log_paths = [
            self.work_dir / "logs" / "conversations.json",
            self.work_dir / "data" / "conversations" / "history.json",
        ]

        for log_path in log_paths:
            if log_path.exists():
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        conversations = json.load(f)

                    # Sample conversations
                    conv_list = conversations if isinstance(conversations, list) else []
                    sampled = random.sample(conv_list, min(num_samples, len(conv_list)))

                    for conv in sampled:
                        samples.append({
                            "text": conv.get("text", str(conv)),
                            "source": "conversations",
                            "metadata": conv.get("metadata", {})
                        })

                    break

                except Exception as e:
                    logger.debug(f"Could not load conversations from {log_path}: {e}")

        await asyncio.sleep(0.1)
        return samples[:num_samples]

    def _save_collection_record(self, metadata: Dict):
        """Save collection record for tracking"""
        record_path = self.data_dir / "collection_history.json"

        self.collection_history.append(metadata)

        try:
            with open(record_path, 'w', encoding='utf-8') as f:
                json.dump(self.collection_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Could not save collection record: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics"""
        return {
            "total_samples_collected": self.collected_samples,
            "total_collections": len(self.collection_history),
            "data_directory": str(self.data_dir),
            "last_collection": self.collection_history[-1] if self.collection_history else None
        }


# For backward compatibility
async def collect_data(target_samples: int, task_type: str = "general") -> Dict[str, List]:
    """Simple function interface for data collection"""
    collector = AutoDataCollector()
    return await collector.collect_training_data(target_samples, task_type)


if __name__ == "__main__":
    # Quick test
    async def test():
        logging.basicConfig(level=logging.INFO)
        collector = AutoDataCollector()

        result = await collector.collect_training_data(100, task_type="general")

        print(f"\n✅ Collected {result['metadata']['total_samples']} samples")
        print(f"   Train: {len(result['train'])}")
        print(f"   Test: {len(result['test'])}")
        print(f"\nSample: {result['train'][0] if result['train'] else 'No data'}")

        stats = collector.get_stats()
        print(f"\nStats: {stats}")

    asyncio.run(test())
