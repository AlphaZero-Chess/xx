"""
Emergent Memory Fusion & Long-Term Cognitive Persistence (Step 31)

This module implements a unified, persistent cognitive memory architecture that:
- Retains insights and reflections beyond individual games
- Fuses strategic, ethical, and creative experiences into long-term adaptive memory
- Enables cross-step recall across creativity (Step 29) and reflection (Step 30)
- Maintains transparency and player access through Human vs AlphaZero interface

Features:
- Memory Fusion Layer integrating short-term reflections with long-term patterns
- Cognitive Persistence Engine for storing/retrieving insights and parameter deltas
- Experience Consolidation distilling learnings into Memory Nodes
- Meta-Evolution Mechanism for adaptive growth based on cumulative memory
- Exponential decay mechanism for memory aging (λ = 0.05)
"""

import logging
import asyncio
import uuid
import math
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
import numpy as np
import os

# Import emergentintegrations for multi-provider LLM support
from emergentintegrations.llm.chat import LlmChat, UserMessage

logger = logging.getLogger(__name__)


@dataclass
class MemoryNode:
    """A single fused memory node representing consolidated learning"""
    memory_id: str
    source_cycles: List[str]  # Reflection cycle IDs that contributed
    key_insight: str
    parameter_delta: Dict[str, float]  # Changes to learning parameters
    ethical_alignment: float  # 0-1
    timestamp: str
    context: Dict[str, Any]  # Game phase, opponent type, etc.
    usage_count: int  # How many times this memory has been retrieved
    last_accessed: str
    decay_weight: float  # Current decay weight (1.0 = full strength)
    confidence_score: float  # 0-1: Confidence in this insight
    
    def to_dict(self):
        return asdict(self)


@dataclass
class PersistenceProfile:
    """Long-term parameter evolution profile"""
    profile_id: str
    timestamp: str
    total_memory_nodes: int
    active_memory_nodes: int  # Not fully decayed
    parameter_evolution: Dict[str, List[float]]  # Parameter history
    creativity_trajectory: List[float]  # Creativity over time
    stability_trajectory: List[float]  # Stability over time
    ethical_trajectory: List[float]  # Ethics over time
    memory_efficiency: float  # How well memory is being utilized
    learning_velocity: float  # Rate of parameter change
    
    def to_dict(self):
        return asdict(self)


@dataclass
class MemoryTrace:
    """Record of memory retrieval and usage"""
    trace_id: str
    timestamp: str
    retrieval_query: str
    memories_retrieved: List[str]  # Memory IDs
    context: Dict[str, Any]
    relevance_scores: List[float]
    usage_type: str  # "game_prep", "reflection", "synthesis"
    
    def to_dict(self):
        return asdict(self)


@dataclass
class MemoryHealthMetrics:
    """Overall memory system health indicators"""
    timestamp: str
    memory_retention_index: float  # 0-1
    fusion_efficiency: float  # 0-1
    ethical_continuity: float  # 0-1
    retrieval_latency: float  # seconds
    persistence_health: float  # 0-1
    total_nodes: int
    active_nodes: int
    decayed_nodes: int
    avg_decay_rate: float
    memory_diversity: float  # Variety of stored insights
    
    def to_dict(self):
        return asdict(self)


class MemoryFusionController:
    """
    Memory Fusion & Long-Term Cognitive Persistence Controller
    
    Orchestrates the fusion of short-term reflections (Step 30) and creative
    strategies (Step 29) into persistent long-term memory nodes with decay.
    """
    
    def __init__(self, db_client):
        self.db = db_client
        self.llm_key = os.environ.get('EMERGENT_LLM_KEY')
        
        if not self.llm_key:
            logger.warning("EMERGENT_LLM_KEY not found - memory fusion will use mock mode")
            self.llm_available = False
        else:
            self.llm_available = True
            logger.info("Memory Fusion Controller initialized with multi-provider LLM support")
        
        # Memory fusion parameters (from problem statement)
        self.max_nodes_per_cycle = 5  # Maximum memory nodes per fusion cycle
        self.retention_window = 30  # Rolling context window in games
        self.decay_lambda = 0.05  # Exponential decay rate
        self.min_decay_threshold = 0.20  # Below this, memory is considered inactive
        self.retrieval_latency_target = 2.0  # Target latency in seconds
        
        # Target metrics (from problem statement)
        self.target_metrics = {
            "memory_retention_index": 0.90,
            "fusion_efficiency": 0.85,
            "ethical_continuity": 0.92,
            "retrieval_latency": 2.0,
            "persistence_health": 0.88
        }
        
        # LLM providers prioritized per problem statement
        self.llm_providers = {
            "primary": {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
            "secondary": {"provider": "openai", "model": "gpt-4o-mini"},
            "fallback": {"provider": "google", "model": "gemini-2.0-flash-exp"}
        }
        
        logger.info(f"Memory Fusion initialized: λ={self.decay_lambda}, max_nodes={self.max_nodes_per_cycle}")
    
    async def trigger_fusion_cycle(
        self,
        reflection_cycle_id: str,
        reflection_data: Dict[str, Any],
        trigger: str = "post_reflection"
    ) -> Dict[str, Any]:
        """
        Trigger a complete memory fusion cycle.
        
        Called automatically after Step 30 reflection cycles to consolidate
        learning into long-term memory nodes.
        
        Args:
            reflection_cycle_id: ID of the reflection cycle that triggered this
            reflection_data: Complete reflection cycle data from Step 30
            trigger: What triggered this fusion
        
        Returns:
            Fusion cycle results with new memory nodes
        """
        try:
            fusion_id = str(uuid.uuid4())
            start_time = datetime.now()
            
            logger.info(f"Starting memory fusion cycle {fusion_id} (trigger: {trigger})")
            
            # Step 1: Apply decay to existing memory nodes
            await self._apply_memory_decay()
            
            # Step 2: Extract insights from reflection cycle
            insights = await self._extract_reflection_insights(reflection_data)
            
            # Step 3: Gather creative strategy performance data
            strategy_data = await self._gather_strategy_performance()
            
            # Step 4: Fuse insights into memory nodes
            new_memory_nodes = await self._fuse_into_memory_nodes(
                insights, strategy_data, reflection_cycle_id
            )
            
            # Step 5: Store memory nodes (limit to max per cycle)
            stored_nodes = []
            for node in new_memory_nodes[:self.max_nodes_per_cycle]:
                await self.db.llm_memory_nodes.insert_one(node.to_dict())
                stored_nodes.append(node)
            
            logger.info(f"Stored {len(stored_nodes)} new memory nodes")
            
            # Step 6: Update persistence profile
            profile = await self._update_persistence_profile(stored_nodes)
            
            # Step 7: Update cognitive weights based on memory
            weight_updates = await self._update_cognitive_weights(stored_nodes)
            
            # Step 8: Calculate fusion metrics
            metrics = await self._calculate_fusion_metrics()
            
            # Calculate elapsed time
            elapsed = (datetime.now() - start_time).total_seconds()
            
            result = {
                "fusion_id": fusion_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trigger": trigger,
                "source_reflection_cycle": reflection_cycle_id,
                "new_memory_nodes": len(stored_nodes),
                "memory_nodes_created": [n.memory_id for n in stored_nodes],
                "persistence_profile_updated": profile.profile_id if profile else None,
                "cognitive_weight_updates": weight_updates,
                "fusion_metrics": metrics.to_dict() if metrics else None,
                "fusion_time_seconds": elapsed,
                "success": True
            }
            
            # Store fusion cycle record
            await self.db.llm_memory_trace.insert_one({
                **result,
                "trace_type": "fusion_cycle"
            })
            
            logger.info(
                f"Memory fusion cycle {fusion_id} complete: "
                f"{len(stored_nodes)} nodes created in {elapsed:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in memory fusion cycle: {e}")
            return {
                "fusion_id": fusion_id if 'fusion_id' in locals() else "error",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _apply_memory_decay(self):
        """Apply exponential decay to all existing memory nodes"""
        try:
            # Get all active memory nodes
            nodes = await self.db.llm_memory_nodes.find({
                "decay_weight": {"$gte": self.min_decay_threshold}
            }).to_list(1000)
            
            if not nodes:
                return
            
            current_time = datetime.now(timezone.utc)
            
            for node in nodes:
                node_id = node["memory_id"]
                created_at = datetime.fromisoformat(node["timestamp"].replace('Z', '+00:00'))
                last_accessed = datetime.fromisoformat(node["last_accessed"].replace('Z', '+00:00'))
                
                # Time since last access (in days)
                time_delta = (current_time - last_accessed).total_seconds() / (24 * 3600)
                
                # Exponential decay: weight = exp(-λ × t)
                new_decay_weight = math.exp(-self.decay_lambda * time_delta)
                
                # Update if changed significantly
                if abs(new_decay_weight - node["decay_weight"]) > 0.01:
                    await self.db.llm_memory_nodes.update_one(
                        {"memory_id": node_id},
                        {"$set": {"decay_weight": new_decay_weight}}
                    )
            
            logger.info(f"Applied decay to {len(nodes)} memory nodes")
            
        except Exception as e:
            logger.error(f"Error applying memory decay: {e}")
    
    async def _extract_reflection_insights(
        self,
        reflection_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract key insights from reflection cycle data"""
        try:
            insights = []
            
            # Extract from game reflections
            game_reflections = reflection_data.get("game_reflections", [])
            for game_reflection in game_reflections:
                # Collect learning insights
                for insight in game_reflection.get("learning_insights", []):
                    insights.append({
                        "type": "game_learning",
                        "insight": insight,
                        "source_game": game_reflection.get("game_id"),
                        "decision_quality": game_reflection.get("decision_quality_score", 0.0),
                        "ethical_score": game_reflection.get("ethical_compliance_score", 0.0),
                        "outcome": game_reflection.get("game_outcome")
                    })
            
            # Extract from strategy evaluations
            strategy_evaluations = reflection_data.get("strategy_evaluations", [])
            for strategy_eval in strategy_evaluations:
                if strategy_eval.get("performance_rating") in ["excellent", "good"]:
                    insights.append({
                        "type": "strategy_performance",
                        "insight": strategy_eval.get("critique", ""),
                        "strategy_name": strategy_eval.get("strategy_name"),
                        "success_rate": strategy_eval.get("success_rate", 0.0),
                        "ethical_score": strategy_eval.get("avg_ethical_score", 0.0)
                    })
            
            # Extract parameter adjustment recommendations
            param_adjustments = reflection_data.get("parameter_adjustments", {})
            if param_adjustments.get("reasoning"):
                insights.append({
                    "type": "parameter_adjustment",
                    "insight": "; ".join(param_adjustments.get("reasoning", [])),
                    "adjustments": param_adjustments
                })
            
            logger.info(f"Extracted {len(insights)} insights from reflection data")
            return insights
            
        except Exception as e:
            logger.error(f"Error extracting reflection insights: {e}")
            return []
    
    async def _gather_strategy_performance(self) -> Dict[str, Any]:
        """Gather recent creative strategy performance data"""
        try:
            # Get recent strategy evaluations from Step 30
            evaluations = await self.db.llm_strategy_evaluation.find().sort(
                "timestamp", -1
            ).limit(20).to_list(20)
            
            # Get creative strategies from Step 29
            strategies = await self.db.llm_creative_synthesis.find({
                "rejected": False
            }).sort("timestamp", -1).limit(10).to_list(10)
            
            return {
                "recent_evaluations": evaluations,
                "active_strategies": strategies,
                "total_evaluated": len(evaluations),
                "total_active": len(strategies)
            }
            
        except Exception as e:
            logger.error(f"Error gathering strategy performance: {e}")
            return {}
    
    async def _fuse_into_memory_nodes(
        self,
        insights: List[Dict[str, Any]],
        strategy_data: Dict[str, Any],
        reflection_cycle_id: str
    ) -> List[MemoryNode]:
        """
        Fuse insights and strategy data into consolidated memory nodes.
        
        Uses LLM to synthesize high-level patterns and meta-learnings.
        """
        try:
            if not insights:
                logger.info("No insights to fuse into memory")
                return []
            
            # Group insights by type for better synthesis
            game_insights = [i for i in insights if i["type"] == "game_learning"]
            strategy_insights = [i for i in insights if i["type"] == "strategy_performance"]
            param_insights = [i for i in insights if i["type"] == "parameter_adjustment"]
            
            memory_nodes = []
            
            # Fuse game learning insights
            if game_insights:
                node = await self._create_memory_node_from_insights(
                    game_insights,
                    "game_learning",
                    reflection_cycle_id,
                    strategy_data
                )
                if node:
                    memory_nodes.append(node)
            
            # Fuse strategy performance insights
            if strategy_insights:
                node = await self._create_memory_node_from_insights(
                    strategy_insights,
                    "strategy_performance",
                    reflection_cycle_id,
                    strategy_data
                )
                if node:
                    memory_nodes.append(node)
            
            # Fuse parameter adjustment insights
            if param_insights:
                node = await self._create_memory_node_from_insights(
                    param_insights,
                    "parameter_adjustment",
                    reflection_cycle_id,
                    strategy_data
                )
                if node:
                    memory_nodes.append(node)
            
            logger.info(f"Fused {len(memory_nodes)} memory nodes from {len(insights)} insights")
            return memory_nodes
            
        except Exception as e:
            logger.error(f"Error fusing into memory nodes: {e}")
            return []
    
    async def _create_memory_node_from_insights(
        self,
        insights: List[Dict[str, Any]],
        insight_type: str,
        reflection_cycle_id: str,
        strategy_data: Dict[str, Any]
    ) -> Optional[MemoryNode]:
        """Create a single memory node from a group of insights"""
        try:
            if self.llm_available:
                key_insight = await self._llm_synthesize_insight(insights, insight_type)
            else:
                key_insight = self._mock_synthesize_insight(insights, insight_type)
            
            # Calculate parameter deltas based on insights
            parameter_delta = self._calculate_parameter_delta(insights, insight_type)
            
            # Calculate ethical alignment
            ethical_scores = [i.get("ethical_score", 0.9) for i in insights if "ethical_score" in i]
            avg_ethical = np.mean(ethical_scores) if ethical_scores else 0.90
            
            # Determine context
            context = self._extract_context_from_insights(insights, insight_type)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(insights, insight_type)
            
            node = MemoryNode(
                memory_id=str(uuid.uuid4()),
                source_cycles=[reflection_cycle_id],
                key_insight=key_insight,
                parameter_delta=parameter_delta,
                ethical_alignment=round(avg_ethical, 2),
                timestamp=datetime.now(timezone.utc).isoformat(),
                context=context,
                usage_count=0,
                last_accessed=datetime.now(timezone.utc).isoformat(),
                decay_weight=1.0,  # Start at full strength
                confidence_score=confidence
            )
            
            return node
            
        except Exception as e:
            logger.error(f"Error creating memory node: {e}")
            return None
    
    async def _llm_synthesize_insight(
        self,
        insights: List[Dict[str, Any]],
        insight_type: str
    ) -> str:
        """Use LLM to synthesize key insight from multiple insights [PROD]"""
        try:
            provider_config = self.llm_providers["primary"]
            
            chat = LlmChat(
                api_key=self.llm_key,
                session_id=f"memory-fusion-{uuid.uuid4()}",
                system_message="You are an AI memory synthesis system consolidating learning insights into persistent memory patterns."
            ).with_model(provider_config["provider"], provider_config["model"])
            
            # Build insights summary
            insights_text = "\n".join([
                f"- {i.get('insight', 'No insight')}" 
                for i in insights[:5]  # Limit to top 5
            ])
            
            type_context = {
                "game_learning": "game performance and decision-making patterns",
                "strategy_performance": "creative strategy effectiveness and viability",
                "parameter_adjustment": "learning parameter optimization recommendations"
            }.get(insight_type, "general learning patterns")
            
            prompt = f"""Synthesize the following {insight_type} insights into a single, concise memory pattern:

**Insights to consolidate:**
{insights_text}

**Context:** These insights relate to {type_context}.

**Your task:**
Create ONE consolidated insight (1-2 sentences) that:
1. Captures the core pattern across all insights
2. Is actionable and specific
3. Maintains ethical considerations
4. Can guide future decision-making

**Output format:**
[Single consolidated insight in 1-2 sentences]

Be concise and focus on the most impactful pattern."""

            user_message = UserMessage(text=prompt)
            response = await chat.send_message(user_message)
            
            synthesized = response.strip()
            logger.info(f"[PROD] LLM synthesized {insight_type} insight")
            
            return synthesized
            
        except Exception as e:
            logger.error(f"Error in LLM insight synthesis: {e}, using fallback")
            return self._mock_synthesize_insight(insights, insight_type)
    
    def _mock_synthesize_insight(
        self,
        insights: List[Dict[str, Any]],
        insight_type: str
    ) -> str:
        """Mock insight synthesis [MOCK]"""
        if insight_type == "game_learning":
            return f"Game performance analysis across {len(insights)} insights reveals consistent patterns in decision quality and strategic execution. Key learning: balanced approach yields optimal outcomes."
        elif insight_type == "strategy_performance":
            return f"Creative strategy evaluation across {len(insights)} applications shows promising results with strong ethical alignment. Recommendation: continue current strategic approaches with incremental refinements."
        elif insight_type == "parameter_adjustment":
            return f"Parameter optimization analysis suggests {len(insights)} key adjustments to enhance learning efficiency while maintaining ethical standards and stability."
        else:
            return f"Consolidated learning pattern synthesized from {len(insights)} insights with focus on continuous improvement and adaptation."
    
    def _calculate_parameter_delta(
        self,
        insights: List[Dict[str, Any]],
        insight_type: str
    ) -> Dict[str, float]:
        """Calculate suggested parameter changes based on insights"""
        deltas = {}
        
        if insight_type == "game_learning":
            # Analyze decision quality trends
            qualities = [i.get("decision_quality", 0.75) for i in insights if "decision_quality" in i]
            if qualities:
                avg_quality = np.mean(qualities)
                if avg_quality > 0.85:
                    deltas["stability_weight"] = +0.02  # Reinforce stability
                elif avg_quality < 0.65:
                    deltas["stability_weight"] = +0.04  # Need more stability
        
        elif insight_type == "strategy_performance":
            # Analyze strategy success rates
            success_rates = [i.get("success_rate", 0.5) for i in insights if "success_rate" in i]
            if success_rates:
                avg_success = np.mean(success_rates)
                if avg_success > 0.70:
                    deltas["novelty_weight"] = +0.03  # Can afford more creativity
                elif avg_success < 0.50:
                    deltas["novelty_weight"] = -0.02  # Reduce creativity, focus stability
        
        elif insight_type == "parameter_adjustment":
            # Extract explicit adjustments
            for insight in insights:
                if "adjustments" in insight:
                    adj = insight["adjustments"]
                    for key in ["novelty_weight_delta", "stability_weight_delta", "ethical_threshold_delta"]:
                        if key in adj and adj[key] != 0:
                            param_name = key.replace("_delta", "")
                            deltas[param_name] = adj[key]
        
        return deltas
    
    def _extract_context_from_insights(
        self,
        insights: List[Dict[str, Any]],
        insight_type: str
    ) -> Dict[str, Any]:
        """Extract contextual information from insights"""
        context = {
            "insight_type": insight_type,
            "insight_count": len(insights),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if insight_type == "game_learning":
            outcomes = [i.get("outcome") for i in insights if "outcome" in i]
            if outcomes:
                context["game_outcomes"] = {
                    "wins": outcomes.count("win"),
                    "losses": outcomes.count("loss"),
                    "draws": outcomes.count("draw")
                }
        
        elif insight_type == "strategy_performance":
            strategies = [i.get("strategy_name") for i in insights if "strategy_name" in i]
            if strategies:
                context["strategies_evaluated"] = list(set(strategies))
        
        return context
    
    def _calculate_confidence_score(
        self,
        insights: List[Dict[str, Any]],
        insight_type: str
    ) -> float:
        """Calculate confidence score for the synthesized insight"""
        # Base confidence on number of supporting insights
        base_confidence = min(0.95, 0.60 + (len(insights) * 0.05))
        
        # Adjust based on insight consistency
        if insight_type == "game_learning":
            qualities = [i.get("decision_quality", 0.75) for i in insights if "decision_quality" in i]
            if qualities:
                std_dev = np.std(qualities)
                consistency_factor = max(0.0, 1.0 - std_dev)
                base_confidence *= (0.7 + 0.3 * consistency_factor)
        
        return round(min(0.99, max(0.50, base_confidence)), 2)
    
    async def _update_persistence_profile(
        self,
        new_nodes: List[MemoryNode]
    ) -> Optional[PersistenceProfile]:
        """Update long-term persistence profile with new memory nodes"""
        try:
            # Get all active memory nodes
            all_nodes = await self.db.llm_memory_nodes.find({
                "decay_weight": {"$gte": self.min_decay_threshold}
            }).to_list(1000)
            
            total_nodes = await self.db.llm_memory_nodes.count_documents({})
            
            # Get previous profile for trajectory tracking
            prev_profile = await self.db.llm_persistence_profile.find_one(
                sort=[("timestamp", -1)]
            )
            
            # Calculate parameter evolution
            param_evolution = {}
            for node in all_nodes:
                for param_name, delta in node.get("parameter_delta", {}).items():
                    if param_name not in param_evolution:
                        param_evolution[param_name] = []
                    param_evolution[param_name].append(delta)
            
            # Calculate trajectories (simplified for now)
            creativity_trajectory = [0.70, 0.72, 0.75]  # Would calculate from actual data
            stability_trajectory = [0.65, 0.68, 0.70]
            ethical_trajectory = [0.90, 0.91, 0.92]
            
            if prev_profile:
                creativity_trajectory = prev_profile.get("creativity_trajectory", [])[-10:] + [0.75]
                stability_trajectory = prev_profile.get("stability_trajectory", [])[-10:] + [0.70]
                ethical_trajectory = prev_profile.get("ethical_trajectory", [])[-10:] + [0.92]
            
            # Calculate memory efficiency
            if total_nodes > 0:
                memory_efficiency = len(all_nodes) / total_nodes
            else:
                memory_efficiency = 1.0
            
            # Calculate learning velocity (rate of parameter change)
            recent_deltas = []
            for node in all_nodes[-5:]:  # Last 5 nodes
                deltas = node.get("parameter_delta", {}).values()
                recent_deltas.extend(deltas)
            
            learning_velocity = np.mean([abs(d) for d in recent_deltas]) if recent_deltas else 0.0
            
            profile = PersistenceProfile(
                profile_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                total_memory_nodes=total_nodes,
                active_memory_nodes=len(all_nodes),
                parameter_evolution=param_evolution,
                creativity_trajectory=creativity_trajectory,
                stability_trajectory=stability_trajectory,
                ethical_trajectory=ethical_trajectory,
                memory_efficiency=round(memory_efficiency, 2),
                learning_velocity=round(learning_velocity, 3)
            )
            
            # Store profile
            await self.db.llm_persistence_profile.insert_one(profile.to_dict())
            
            logger.info(f"Persistence profile updated: {len(all_nodes)} active nodes")
            return profile
            
        except Exception as e:
            logger.error(f"Error updating persistence profile: {e}")
            return None
    
    async def _update_cognitive_weights(
        self,
        new_nodes: List[MemoryNode]
    ) -> Dict[str, Any]:
        """Update cognitive weights based on new memory nodes"""
        try:
            # Aggregate parameter deltas from new nodes
            total_deltas = {}
            for node in new_nodes:
                for param, delta in node.parameter_delta.items():
                    if param not in total_deltas:
                        total_deltas[param] = []
                    # Weight by confidence and decay
                    weighted_delta = delta * node.confidence_score * node.decay_weight
                    total_deltas[param].append(weighted_delta)
            
            # Calculate average deltas
            avg_deltas = {}
            for param, deltas in total_deltas.items():
                avg_deltas[param] = round(np.mean(deltas), 4) if deltas else 0.0
            
            updates = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_nodes": [n.memory_id for n in new_nodes],
                "parameter_updates": avg_deltas,
                "applied": False,  # Advisory mode only
                "recommendation": "Review and apply manually through reflection parameters"
            }
            
            logger.info(f"Cognitive weight updates calculated: {avg_deltas}")
            return updates
            
        except Exception as e:
            logger.error(f"Error updating cognitive weights: {e}")
            return {}
    
    async def _calculate_fusion_metrics(self) -> Optional[MemoryHealthMetrics]:
        """Calculate comprehensive memory system health metrics"""
        try:
            # Get all nodes
            all_nodes = await self.db.llm_memory_nodes.find().to_list(1000)
            active_nodes = [n for n in all_nodes if n.get("decay_weight", 0) >= self.min_decay_threshold]
            decayed_nodes = [n for n in all_nodes if n.get("decay_weight", 1.0) < self.min_decay_threshold]
            
            # Calculate memory retention index
            if all_nodes:
                memory_retention = len(active_nodes) / len(all_nodes)
            else:
                memory_retention = 1.0
            
            # Calculate fusion efficiency (how well insights are being converted)
            recent_cycles = await self.db.llm_memory_trace.find({
                "trace_type": "fusion_cycle"
            }).sort("timestamp", -1).limit(10).to_list(10)
            
            if recent_cycles:
                avg_nodes_per_cycle = np.mean([c.get("new_memory_nodes", 0) for c in recent_cycles])
                fusion_efficiency = min(1.0, avg_nodes_per_cycle / self.max_nodes_per_cycle)
            else:
                fusion_efficiency = 0.0
            
            # Calculate ethical continuity
            if active_nodes:
                ethical_scores = [n.get("ethical_alignment", 0.9) for n in active_nodes]
                ethical_continuity = np.mean(ethical_scores)
            else:
                ethical_continuity = 0.9
            
            # Calculate retrieval latency (from recent traces)
            recent_retrievals = await self.db.llm_memory_trace.find({
                "trace_type": "retrieval"
            }).sort("timestamp", -1).limit(20).to_list(20)
            
            if recent_retrievals:
                latencies = [r.get("retrieval_time", 1.0) for r in recent_retrievals if "retrieval_time" in r]
                retrieval_latency = np.mean(latencies) if latencies else 1.0
            else:
                retrieval_latency = 0.5  # Optimistic default
            
            # Calculate persistence health (weighted combination)
            persistence_health = (
                memory_retention * 0.30 +
                fusion_efficiency * 0.25 +
                ethical_continuity * 0.25 +
                (1.0 - min(1.0, retrieval_latency / self.retrieval_latency_target)) * 0.20
            )
            
            # Calculate average decay rate
            if active_nodes:
                decay_rates = [1.0 - n.get("decay_weight", 1.0) for n in active_nodes]
                avg_decay_rate = np.mean(decay_rates)
            else:
                avg_decay_rate = 0.0
            
            # Calculate memory diversity (unique insight types)
            if all_nodes:
                insight_types = set([n.get("context", {}).get("insight_type", "unknown") for n in all_nodes])
                memory_diversity = len(insight_types) / 3.0  # Normalize by expected types
            else:
                memory_diversity = 0.0
            
            metrics = MemoryHealthMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                memory_retention_index=round(memory_retention, 2),
                fusion_efficiency=round(fusion_efficiency, 2),
                ethical_continuity=round(ethical_continuity, 2),
                retrieval_latency=round(retrieval_latency, 2),
                persistence_health=round(persistence_health, 2),
                total_nodes=len(all_nodes),
                active_nodes=len(active_nodes),
                decayed_nodes=len(decayed_nodes),
                avg_decay_rate=round(avg_decay_rate, 3),
                memory_diversity=round(min(1.0, memory_diversity), 2)
            )
            
            logger.info(
                f"Memory health metrics: "
                f"Retention={metrics.memory_retention_index:.2f}, "
                f"Health={metrics.persistence_health:.2f}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating fusion metrics: {e}")
            return None
    
    async def retrieve_contextual_memory(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        limit: int = 5
    ) -> List[MemoryNode]:
        """
        Retrieve relevant memory nodes based on contextual query.
        
        Args:
            query: Context or question to retrieve memories for
            context: Additional contextual filters (game phase, etc.)
            limit: Maximum memories to retrieve
        
        Returns:
            List of relevant memory nodes, ranked by relevance
        """
        try:
            start_time = datetime.now()
            
            # Get all active memory nodes
            active_nodes = await self.db.llm_memory_nodes.find({
                "decay_weight": {"$gte": self.min_decay_threshold}
            }).sort("decay_weight", -1).limit(50).to_list(50)
            
            if not active_nodes:
                logger.info("No active memory nodes to retrieve")
                return []
            
            # Simple relevance scoring (in production would use embeddings)
            scored_nodes = []
            query_lower = query.lower()
            
            for node_doc in active_nodes:
                # Reconstruct MemoryNode object
                node = MemoryNode(**{k: v for k, v in node_doc.items() if k != '_id'})
                
                # Calculate relevance score
                relevance = 0.0
                
                # Text matching
                if query_lower in node.key_insight.lower():
                    relevance += 0.5
                
                # Context matching
                if context:
                    node_context = node.context
                    for key, value in context.items():
                        if node_context.get(key) == value:
                            relevance += 0.2
                
                # Decay weight contribution
                relevance += node.decay_weight * 0.3
                
                # Confidence contribution
                relevance += node.confidence_score * 0.2
                
                scored_nodes.append((node, relevance))
            
            # Sort by relevance
            scored_nodes.sort(key=lambda x: x[1], reverse=True)
            
            # Get top nodes
            top_nodes = [node for node, score in scored_nodes[:limit]]
            
            # Update usage counts and last accessed
            for node in top_nodes:
                await self.db.llm_memory_nodes.update_one(
                    {"memory_id": node.memory_id},
                    {
                        "$inc": {"usage_count": 1},
                        "$set": {"last_accessed": datetime.now(timezone.utc).isoformat()}
                    }
                )
            
            # Calculate elapsed time
            elapsed = (datetime.now() - start_time).total_seconds()
            
            # Log retrieval trace
            trace = MemoryTrace(
                trace_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                retrieval_query=query,
                memories_retrieved=[n.memory_id for n in top_nodes],
                context=context or {},
                relevance_scores=[score for _, score in scored_nodes[:limit]],
                usage_type="contextual_query"
            )
            
            await self.db.llm_memory_trace.insert_one({
                **trace.to_dict(),
                "trace_type": "retrieval",
                "retrieval_time": elapsed
            })
            
            logger.info(f"Retrieved {len(top_nodes)} memories in {elapsed:.2f}s for query: '{query[:50]}...'")
            
            return top_nodes
            
        except Exception as e:
            logger.error(f"Error retrieving contextual memory: {e}")
            return []
    
    async def synthesize_long_term_profile(self) -> Dict[str, Any]:
        """
        Synthesize a high-level summary of long-term memory and learning trends.
        
        Returns:
            Comprehensive profile of memory system evolution
        """
        try:
            # Get latest persistence profile
            latest_profile = await self.db.llm_persistence_profile.find_one(
                sort=[("timestamp", -1)]
            )
            
            # Get memory health metrics
            metrics = await self._calculate_fusion_metrics()
            
            # Get all active memory nodes
            active_nodes = await self.db.llm_memory_nodes.find({
                "decay_weight": {"$gte": self.min_decay_threshold}
            }).to_list(1000)
            
            # Analyze memory distribution
            memory_distribution = {
                "game_learning": 0,
                "strategy_performance": 0,
                "parameter_adjustment": 0,
                "other": 0
            }
            
            for node in active_nodes:
                insight_type = node.get("context", {}).get("insight_type", "other")
                if insight_type in memory_distribution:
                    memory_distribution[insight_type] += 1
                else:
                    memory_distribution["other"] += 1
            
            # Calculate learning trends
            if latest_profile:
                creativity_trend = self._calculate_trend(latest_profile.get("creativity_trajectory", []))
                stability_trend = self._calculate_trend(latest_profile.get("stability_trajectory", []))
                ethical_trend = self._calculate_trend(latest_profile.get("ethical_trajectory", []))
            else:
                creativity_trend = stability_trend = ethical_trend = "stable"
            
            # Generate insights summary
            if self.llm_available:
                summary = await self._llm_generate_profile_summary(
                    active_nodes, metrics, latest_profile
                )
            else:
                summary = self._mock_generate_profile_summary(
                    len(active_nodes), metrics
                )
            
            profile = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_active_memories": len(active_nodes),
                "memory_distribution": memory_distribution,
                "learning_trends": {
                    "creativity": creativity_trend,
                    "stability": stability_trend,
                    "ethical_alignment": ethical_trend
                },
                "health_metrics": metrics.to_dict() if metrics else None,
                "persistence_profile": latest_profile,
                "memory_efficiency": latest_profile.get("memory_efficiency") if latest_profile else 0.0,
                "learning_velocity": latest_profile.get("learning_velocity") if latest_profile else 0.0,
                "summary": summary,
                "target_comparison": {
                    "memory_retention": {
                        "current": metrics.memory_retention_index if metrics else 0,
                        "target": self.target_metrics["memory_retention_index"],
                        "status": "✅" if (metrics and metrics.memory_retention_index >= self.target_metrics["memory_retention_index"]) else "⚠️"
                    },
                    "fusion_efficiency": {
                        "current": metrics.fusion_efficiency if metrics else 0,
                        "target": self.target_metrics["fusion_efficiency"],
                        "status": "✅" if (metrics and metrics.fusion_efficiency >= self.target_metrics["fusion_efficiency"]) else "⚠️"
                    },
                    "ethical_continuity": {
                        "current": metrics.ethical_continuity if metrics else 0,
                        "target": self.target_metrics["ethical_continuity"],
                        "status": "✅" if (metrics and metrics.ethical_continuity >= self.target_metrics["ethical_continuity"]) else "⚠️"
                    },
                    "persistence_health": {
                        "current": metrics.persistence_health if metrics else 0,
                        "target": self.target_metrics["persistence_health"],
                        "status": "✅" if (metrics and metrics.persistence_health >= self.target_metrics["persistence_health"]) else "⚠️"
                    }
                }
            }
            
            logger.info("Long-term profile synthesized successfully")
            return profile
            
        except Exception as e:
            logger.error(f"Error synthesizing long-term profile: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _calculate_trend(self, trajectory: List[float]) -> str:
        """Calculate trend direction from trajectory"""
        if not trajectory or len(trajectory) < 2:
            return "stable"
        
        recent = trajectory[-3:]  # Last 3 points
        if len(recent) < 2:
            return "stable"
        
        delta = recent[-1] - recent[0]
        
        if delta > 0.05:
            return "improving"
        elif delta < -0.05:
            return "declining"
        else:
            return "stable"
    
    async def _llm_generate_profile_summary(
        self,
        active_nodes: List[Dict],
        metrics: Optional[MemoryHealthMetrics],
        profile: Optional[Dict]
    ) -> str:
        """Generate LLM summary of long-term profile [PROD]"""
        try:
            provider_config = self.llm_providers["primary"]
            
            chat = LlmChat(
                api_key=self.llm_key,
                session_id=f"profile-summary-{uuid.uuid4()}",
                system_message="You are an AI memory analyst summarizing long-term cognitive evolution patterns."
            ).with_model(provider_config["provider"], provider_config["model"])
            
            prompt = f"""Generate a concise summary (3-4 sentences) of the AI's long-term memory and learning evolution:

**Memory Statistics:**
- Active Memory Nodes: {len(active_nodes)}
- Memory Retention Index: {metrics.memory_retention_index if metrics else 'N/A'}
- Fusion Efficiency: {metrics.fusion_efficiency if metrics else 'N/A'}
- Ethical Continuity: {metrics.ethical_continuity if metrics else 'N/A'}
- Persistence Health: {metrics.persistence_health if metrics else 'N/A'}

**Learning Velocity:** {profile.get('learning_velocity') if profile else 'N/A'}
**Memory Efficiency:** {profile.get('memory_efficiency') if profile else 'N/A'}

Summarize:
1. Overall memory system health and performance
2. Key learning patterns or trends observed
3. Areas of strength or areas needing attention
4. Forward-looking insight on cognitive development

Keep it under 100 words and actionable."""

            user_message = UserMessage(text=prompt)
            response = await chat.send_message(user_message)
            
            summary = response.strip()
            logger.info("[PROD] LLM generated long-term profile summary")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating LLM profile summary: {e}")
            return self._mock_generate_profile_summary(len(active_nodes), metrics)
    
    def _mock_generate_profile_summary(
        self,
        node_count: int,
        metrics: Optional[MemoryHealthMetrics]
    ) -> str:
        """Generate mock profile summary [MOCK]"""
        health = metrics.persistence_health if metrics else 0.75
        
        if health >= 0.85:
            status = "excellent health with strong"
        elif health >= 0.70:
            status = "good health with consistent"
        else:
            status = "developing health with emerging"
        
        return (
            f"Long-term memory system operating at {status} retention of learning patterns. "
            f"{node_count} active memory nodes maintain cognitive persistence across game experiences. "
            f"Learning velocity indicates {'rapid' if health >= 0.80 else 'steady'} adaptation to new strategies. "
            f"Ethical alignment remains strong throughout memory evolution. "
            f"Continue current trajectory with regular consolidation cycles for optimal cognitive growth."
        )
    
    async def reset_memory_system(
        self,
        confirmation: str,
        admin_override: bool = False
    ) -> Dict[str, Any]:
        """
        Reset memory system (admin function with safeguards).
        
        Args:
            confirmation: Must be "CONFIRM_RESET"
            admin_override: Admin authorization flag
        
        Returns:
            Reset operation result
        """
        try:
            if confirmation != "CONFIRM_RESET":
                return {
                    "success": False,
                    "error": "Invalid confirmation code",
                    "message": "Memory reset requires confirmation code: CONFIRM_RESET"
                }
            
            if not admin_override:
                return {
                    "success": False,
                    "error": "Admin authorization required",
                    "message": "Memory reset requires admin_override=True"
                }
            
            # Backup before reset
            backup_id = str(uuid.uuid4())
            
            # Get all nodes for backup
            all_nodes = await self.db.llm_memory_nodes.find().to_list(10000)
            all_profiles = await self.db.llm_persistence_profile.find().to_list(1000)
            all_traces = await self.db.llm_memory_trace.find().to_list(10000)
            
            # Store backup
            await self.db.llm_memory_backups.insert_one({
                "backup_id": backup_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "nodes_count": len(all_nodes),
                "profiles_count": len(all_profiles),
                "traces_count": len(all_traces),
                "nodes": all_nodes,
                "profiles": all_profiles,
                "traces": all_traces
            })
            
            # Clear collections
            await self.db.llm_memory_nodes.delete_many({})
            await self.db.llm_persistence_profile.delete_many({})
            await self.db.llm_memory_trace.delete_many({})
            
            logger.warning(f"Memory system reset completed. Backup ID: {backup_id}")
            
            return {
                "success": True,
                "message": "Memory system reset successfully",
                "backup_id": backup_id,
                "nodes_cleared": len(all_nodes),
                "profiles_cleared": len(all_profiles),
                "traces_cleared": len(all_traces),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error resetting memory system: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def get_memory_health(self) -> Dict[str, Any]:
        """Get current memory system health status"""
        try:
            metrics = await self._calculate_fusion_metrics()
            
            if not metrics:
                return {
                    "status": "error",
                    "message": "Unable to calculate health metrics"
                }
            
            # Determine overall status
            health_score = metrics.persistence_health
            
            if health_score >= 0.85:
                status = "excellent"
                emoji = "✅"
            elif health_score >= 0.70:
                status = "good"
                emoji = "✅"
            elif health_score >= 0.50:
                status = "moderate"
                emoji = "⚠️"
            else:
                status = "needs_attention"
                emoji = "🚨"
            
            return {
                "status": status,
                "emoji": emoji,
                "health_score": health_score,
                "metrics": metrics.to_dict(),
                "recommendations": self._generate_health_recommendations(metrics),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting memory health: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _generate_health_recommendations(
        self,
        metrics: MemoryHealthMetrics
    ) -> List[str]:
        """Generate health recommendations based on metrics"""
        recommendations = []
        
        if metrics.memory_retention_index < self.target_metrics["memory_retention_index"]:
            recommendations.append(
                f"Memory retention ({metrics.memory_retention_index:.2f}) below target "
                f"({self.target_metrics['memory_retention_index']:.2f}). "
                "Consider adjusting decay rate or increasing retrieval frequency."
            )
        
        if metrics.fusion_efficiency < self.target_metrics["fusion_efficiency"]:
            recommendations.append(
                f"Fusion efficiency ({metrics.fusion_efficiency:.2f}) below target "
                f"({self.target_metrics['fusion_efficiency']:.2f}). "
                "Ensure reflection cycles are generating sufficient insights."
            )
        
        if metrics.ethical_continuity < self.target_metrics["ethical_continuity"]:
            recommendations.append(
                f"Ethical continuity ({metrics.ethical_continuity:.2f}) below target "
                f"({self.target_metrics['ethical_continuity']:.2f}). "
                "Review ethical alignment in recent memory nodes."
            )
        
        if metrics.retrieval_latency > self.target_metrics["retrieval_latency"]:
            recommendations.append(
                f"Retrieval latency ({metrics.retrieval_latency:.2f}s) exceeds target "
                f"({self.target_metrics['retrieval_latency']:.2f}s). "
                "Consider optimizing retrieval queries or indexing."
            )
        
        if metrics.persistence_health < self.target_metrics["persistence_health"]:
            recommendations.append(
                f"Overall persistence health ({metrics.persistence_health:.2f}) below target "
                f"({self.target_metrics['persistence_health']:.2f}). "
                "Review all subsystem metrics and address specific areas."
            )
        
        if not recommendations:
            recommendations.append(
                "✅ All memory health metrics meeting or exceeding targets. System operating optimally."
            )
        
        return recommendations
