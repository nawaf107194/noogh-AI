#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠+ Enhanced Brain Components - مكونات الدماغ المحسّنة
Additional components to enhance the main brain system
"""

import logging
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, deque
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Knowledge Graph for storing and traversing relationships.

    Features:
    - Add nodes with metadata
    - Create weighted edges
    - Find related nodes with BFS/DFS
    - Path finding
    - Centrality analysis
    - Persistence (save/load)
    """

    def __init__(self, graph_dir: Optional[str] = None):
        """
        Initialize Knowledge Graph.

        Args:
            graph_dir: Directory to save/load graph data
        """
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, List[Dict[str, Any]]] = {}
        self.reverse_edges: Dict[str, List[str]] = defaultdict(list)

        self.graph_dir = Path(graph_dir) if graph_dir else Path("data/knowledge_graph")
        self.graph_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📊 KnowledgeGraph initialized: {self.graph_dir}")

    def add_node(self, node_id: str, data: Dict[str, Any]) -> None:
        """
        Add a node to the graph.

        Args:
            node_id: Unique node identifier
            data: Node metadata dictionary
        """
        self.nodes[node_id] = {
            'id': node_id,
            'data': data,
            'created_at': data.get('created_at', 'unknown')
        }
        logger.debug(f"Added node: {node_id}")

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        weight: float = 1.0,
        edge_type: str = "related"
    ) -> None:
        """
        Add a directed edge between two nodes.

        Args:
            from_id: Source node ID
            to_id: Target node ID
            weight: Edge weight (strength of relationship)
            edge_type: Type of relationship
        """
        # Ensure nodes exist
        if from_id not in self.nodes:
            self.add_node(from_id, {'name': from_id})
        if to_id not in self.nodes:
            self.add_node(to_id, {'name': to_id})

        # Add forward edge
        if from_id not in self.edges:
            self.edges[from_id] = []

        self.edges[from_id].append({
            "to": to_id,
            "weight": weight,
            "type": edge_type
        })

        # Add reverse edge for traversal
        self.reverse_edges[to_id].append(from_id)

        logger.debug(f"Added edge: {from_id} -> {to_id} (weight={weight}, type={edge_type})")

    def find_related(
        self,
        node_id: str,
        max_depth: int = 2,
        min_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find related nodes using BFS traversal.

        Args:
            node_id: Starting node ID
            max_depth: Maximum traversal depth
            min_weight: Minimum edge weight to follow

        Returns:
            List of related nodes with distance and path
        """
        if node_id not in self.nodes:
            logger.warning(f"Node not found: {node_id}")
            return []

        related = []
        visited = {node_id}
        queue = deque([(node_id, 0, [node_id])])  # (node, depth, path)

        while queue:
            current, depth, path = queue.popleft()

            if depth >= max_depth:
                continue

            # Get outgoing edges
            for edge in self.edges.get(current, []):
                neighbor = edge['to']
                weight = edge['weight']
                edge_type = edge.get('type', 'related')

                if neighbor not in visited and weight >= min_weight:
                    visited.add(neighbor)
                    new_path = path + [neighbor]

                    related.append({
                        'node_id': neighbor,
                        'node_data': self.nodes.get(neighbor, {}),
                        'distance': depth + 1,
                        'weight': weight,
                        'type': edge_type,
                        'path': new_path
                    })

                    queue.append((neighbor, depth + 1, new_path))

        # Sort by distance then weight
        related.sort(key=lambda x: (x['distance'], -x['weight']))

        logger.info(f"Found {len(related)} related nodes for {node_id}")
        return related

    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5
    ) -> Optional[List[str]]:
        """
        Find shortest path between two nodes using BFS.

        Args:
            start_id: Starting node ID
            end_id: Target node ID
            max_depth: Maximum search depth

        Returns:
            List of node IDs forming the path, or None
        """
        if start_id not in self.nodes or end_id not in self.nodes:
            return None

        visited = {start_id}
        queue = deque([(start_id, [start_id], 0)])

        while queue:
            current, path, depth = queue.popleft()

            if current == end_id:
                return path

            if depth >= max_depth:
                continue

            for edge in self.edges.get(current, []):
                neighbor = edge['to']
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor], depth + 1))

        return None

    def get_node_degree(self, node_id: str) -> Tuple[int, int]:
        """
        Get in-degree and out-degree of a node.

        Returns:
            Tuple of (in_degree, out_degree)
        """
        out_degree = len(self.edges.get(node_id, []))
        in_degree = len(self.reverse_edges.get(node_id, []))
        return (in_degree, out_degree)

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        total_edges = sum(len(edges) for edges in self.edges.values())

        # Find most connected nodes
        node_degrees = [(nid, sum(self.get_node_degree(nid)))
                       for nid in self.nodes.keys()]
        top_nodes = sorted(node_degrees, key=lambda x: x[1], reverse=True)[:5]

        return {
            "nodes": len(self.nodes),
            "edges": total_edges,
            "avg_degree": total_edges / len(self.nodes) if self.nodes else 0,
            "top_connected_nodes": [{'id': nid, 'degree': deg} for nid, deg in top_nodes]
        }

    def save(self, filename: str = "knowledge_graph.pkl") -> None:
        """Save graph to file."""
        filepath = self.graph_dir / filename
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'nodes': self.nodes,
                    'edges': self.edges,
                    'reverse_edges': dict(self.reverse_edges)
                }, f)
            logger.info(f"✅ Graph saved: {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to save graph: {e}")

    def load(self, filename: str = "knowledge_graph.pkl") -> bool:
        """Load graph from file."""
        filepath = self.graph_dir / filename
        try:
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                self.nodes = data['nodes']
                self.edges = data['edges']
                self.reverse_edges = defaultdict(list, data['reverse_edges'])
                logger.info(f"✅ Graph loaded: {filepath} ({len(self.nodes)} nodes)")
                return True
            else:
                logger.warning(f"Graph file not found: {filepath}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to load graph: {e}")
            return False


class VectorStore:
    """
    Simple Vector Store for semantic similarity search.

    Features:
    - Store vectors with metadata
    - Cosine similarity search
    - Add/remove vectors
    - Persistence
    """

    def __init__(self, vector_dir: Optional[str] = None, dimension: int = 384):
        """
        Initialize Vector Store.

        Args:
            vector_dir: Directory to save/load vectors
            dimension: Vector dimension
        """
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.dimension = dimension

        self.vector_dir = Path(vector_dir) if vector_dir else Path("data/vectors")
        self.vector_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"🔢 VectorStore initialized: dim={dimension}, dir={self.vector_dir}")

    def add_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a vector to the store.

        Args:
            vector_id: Unique vector identifier
            vector: Vector values
            metadata: Optional metadata dictionary
        """
        if len(vector) != self.dimension:
            logger.warning(f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}")
            # Pad or truncate
            if len(vector) < self.dimension:
                vector = vector + [0.0] * (self.dimension - len(vector))
            else:
                vector = vector[:self.dimension]

        self.vectors[vector_id] = vector
        self.metadata[vector_id] = metadata or {}
        logger.debug(f"Added vector: {vector_id}")

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        min_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using cosine similarity.

        Args:
            query_vector: Query vector
            top_k: Number of results to return
            min_score: Minimum similarity score

        Returns:
            List of results with vector_id, score, and metadata
        """
        if len(query_vector) != self.dimension:
            logger.warning(f"Query vector dimension mismatch")
            return []

        results = []

        for vec_id, vec in self.vectors.items():
            score = self._cosine_similarity(query_vector, vec)

            if score >= min_score:
                results.append({
                    'vector_id': vec_id,
                    'score': score,
                    'metadata': self.metadata.get(vec_id, {})
                })

        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)

        return results[:top_k]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get_vector(self, vector_id: str) -> Optional[List[float]]:
        """Get a vector by ID."""
        return self.vectors.get(vector_id)

    def remove_vector(self, vector_id: str) -> bool:
        """Remove a vector from the store."""
        if vector_id in self.vectors:
            del self.vectors[vector_id]
            self.metadata.pop(vector_id, None)
            logger.debug(f"Removed vector: {vector_id}")
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "total_vectors": len(self.vectors),
            "dimension": self.dimension,
            "storage_dir": str(self.vector_dir)
        }

    def save(self, filename: str = "vector_store.pkl") -> None:
        """Save vector store to file."""
        filepath = self.vector_dir / filename
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'vectors': self.vectors,
                    'metadata': self.metadata,
                    'dimension': self.dimension
                }, f)
            logger.info(f"✅ Vector store saved: {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to save vector store: {e}")

    def load(self, filename: str = "vector_store.pkl") -> bool:
        """Load vector store from file."""
        filepath = self.vector_dir / filename
        try:
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                self.vectors = data['vectors']
                self.metadata = data['metadata']
                self.dimension = data['dimension']
                logger.info(f"✅ Vector store loaded: {filepath} ({len(self.vectors)} vectors)")
                return True
            else:
                logger.warning(f"Vector store file not found: {filepath}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to load vector store: {e}")
            return False


class ReasoningEngine:
    """
    Simple Reasoning Engine for logical inference.

    Features:
    - Forward chaining inference
    - Rule-based reasoning
    - Fact management
    - Confidence scoring
    """

    def __init__(self):
        """Initialize Reasoning Engine."""
        self.facts: Set[str] = set()
        self.rules: List[Dict[str, Any]] = []
        self.inference_count = 0

        logger.info("🤔 ReasoningEngine initialized")

    def add_fact(self, fact: str) -> None:
        """Add a fact to the knowledge base."""
        self.facts.add(fact)
        logger.debug(f"Added fact: {fact}")

    def add_rule(
        self,
        conditions: List[str],
        conclusion: str,
        confidence: float = 1.0
    ) -> None:
        """
        Add an inference rule.

        Args:
            conditions: List of conditions (all must be true)
            conclusion: Conclusion to infer
            confidence: Rule confidence (0-1)
        """
        self.rules.append({
            'conditions': conditions,
            'conclusion': conclusion,
            'confidence': confidence
        })
        logger.debug(f"Added rule: {conditions} -> {conclusion} ({confidence})")

    def infer(self, max_iterations: int = 10) -> List[Tuple[str, float]]:
        """
        Perform forward chaining inference.

        Args:
            max_iterations: Maximum inference iterations

        Returns:
            List of (new_fact, confidence) tuples
        """
        new_facts = []

        for iteration in range(max_iterations):
            iteration_new = False

            for rule in self.rules:
                conditions = rule['conditions']
                conclusion = rule['conclusion']
                confidence = rule['confidence']

                # Check if all conditions are met
                if all(cond in self.facts for cond in conditions):
                    # Check if conclusion is new
                    if conclusion not in self.facts:
                        self.facts.add(conclusion)
                        new_facts.append((conclusion, confidence))
                        iteration_new = True
                        self.inference_count += 1
                        logger.debug(f"Inferred: {conclusion} (confidence={confidence})")

            # Stop if no new facts were inferred
            if not iteration_new:
                break

        logger.info(f"Inference complete: {len(new_facts)} new facts in {iteration + 1} iterations")
        return new_facts

    def query(self, fact: str) -> bool:
        """Check if a fact exists in the knowledge base."""
        return fact in self.facts

    def get_stats(self) -> Dict[str, Any]:
        """Get reasoning engine statistics."""
        return {
            "total_facts": len(self.facts),
            "total_rules": len(self.rules),
            "inferences_made": self.inference_count
        }

    def clear(self) -> None:
        """Clear all facts and rules."""
        self.facts.clear()
        self.rules.clear()
        self.inference_count = 0
        logger.info("Reasoning engine cleared")


if __name__ == "__main__":
    # Quick tests
    logging.basicConfig(level=logging.INFO)

    print("\n🧪 Testing Enhanced Brain Components...")

    # Test Knowledge Graph
    print("\n📊 Testing KnowledgeGraph...")
    kg = KnowledgeGraph()
    kg.add_node("AI", {"name": "Artificial Intelligence"})
    kg.add_node("ML", {"name": "Machine Learning"})
    kg.add_node("DL", {"name": "Deep Learning"})
    kg.add_edge("AI", "ML", weight=0.9, edge_type="contains")
    kg.add_edge("ML", "DL", weight=0.8, edge_type="contains")

    related = kg.find_related("AI", max_depth=2)
    print(f"   Found {len(related)} related nodes to AI")
    print(f"   Stats: {kg.get_stats()}")

    # Test Vector Store
    print("\n🔢 Testing VectorStore...")
    vs = VectorStore(dimension=3)
    vs.add_vector("vec1", [1.0, 0.0, 0.0], {"name": "Vector 1"})
    vs.add_vector("vec2", [0.9, 0.1, 0.0], {"name": "Vector 2"})
    vs.add_vector("vec3", [0.0, 1.0, 0.0], {"name": "Vector 3"})

    results = vs.search([1.0, 0.0, 0.0], top_k=2)
    print(f"   Found {len(results)} similar vectors")
    print(f"   Stats: {vs.get_stats()}")

    # Test Reasoning Engine
    print("\n🤔 Testing ReasoningEngine...")
    re = ReasoningEngine()
    re.add_fact("mammal(cat)")
    re.add_fact("mammal(dog)")
    re.add_rule(["mammal(cat)"], "has_fur(cat)", confidence=0.95)
    re.add_rule(["mammal(dog)"], "has_fur(dog)", confidence=0.95)

    new_facts = re.infer()
    print(f"   Inferred {len(new_facts)} new facts")
    print(f"   Stats: {re.get_stats()}")

    print("\n✅ All tests complete!")
