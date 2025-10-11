"""
Autonomous Cognitive Synthesis & Value Preservation Layer (Step 27)

This module implements meta-adaptive cognitive synthesis that integrates insights
from all prior layers (Steps 19-26), preserves long-term value alignment, and
maintains ethical stability while optimizing for performance.
"""

import logging
import asyncio
import uuid
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
import numpy as np
import os

# Import emergentintegrations for LLM support
from emergentintegrations.llm.chat import LlmChat, UserMessage

logger = logging.getLogger(__name__)


@dataclass
class MultilayerInsight:
    """Insight from a specific system layer"""
    layer_name: str  # "memory", "intelligence", "optimization", "governance", "consensus"
    insight_type: str
    content: str
    confidence: float  # 0-1
    timestamp: str
    metrics: Dict[str, float]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class CognitivePattern:
    """Emergent cognitive pattern detected across layers"""
    pattern_id: str
    pattern_name: str
    description: str
    layers_involved: List[str]
    strength: float  # 0-1
    emergence_count: int
    first_detected: str
    last_detected: str
    impact_areas: List[str]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ValueState:
    """Current state of a system value"""
    value_name: str
    category: str  # "ethical", "strategic", "performance"
    current_score: float  # 0-100
    target_score: float  # 0-100
    drift_amount: float  # Change from baseline
    drift_direction: str  # "positive", "negative", "stable"
    stability_index: float  # 0-1
    last_updated: str
    
    def to_dict(self):
        return asdict(self)


@dataclass
class SynthesisCycle:
    """Complete cognitive synthesis cycle result"""
    cycle_id: str
    timestamp: str
    trigger: str  # "scheduled", "manual", "threshold"
    layers_integrated: List[str]
    insights_collected: int
    patterns_detected: List[CognitivePattern]
    value_states: List[ValueState]
    drift_report: Dict[str, Any]
    stability_metrics: Dict[str, float]
    reflection_summary: str
    cognitive_coherence_index: float  # 0-1
    value_integrity_score: float  # 0-100
    recommendations: List[str]
    
    def to_dict(self):
        result = asdict(self)
        result['patterns_detected'] = [p.to_dict() if hasattr(p, 'to_dict') else p 
                                       for p in result.get('patterns_detected', [])]
        result['value_states'] = [v.to_dict() if hasattr(v, 'to_dict') else v 
                                  for v in result.get('value_states', [])]
        return result


class CognitiveSynthesisController:
    """
    Autonomous cognitive synthesis controller that integrates insights from all
    system layers and preserves long-term value alignment.
    """
    
    def __init__(self, db_client):
        self.db = db_client
        self.llm_key = os.environ.get('EMERGENT_LLM_KEY')
        
        if not self.llm_key:
            logger.warning("EMERGENT_LLM_KEY not found - using sample data mode")
            self.llm_available = False
        else:
            self.llm_available = True
            logger.info("Cognitive Synthesis Controller initialized with LLM support")
        
        # Core system values to monitor
        self.core_values = {
            "transparency": {"target": 85.0, "category": "ethical"},
            "fairness": {"target": 88.0, "category": "ethical"},
            "safety": {"target": 92.0, "category": "ethical"},
            "performance": {"target": 78.0, "category": "performance"},
            "alignment": {"target": 85.0, "category": "strategic"},
            "stability": {"target": 90.0, "category": "strategic"}
        }
        
        # Value baselines (initialized on first synthesis)
        self.value_baselines = {}
        
        # Pattern detection thresholds
        self.pattern_strength_threshold = 0.65
        self.pattern_emergence_threshold = 3
    
    async def aggregate_multilayer_insights(self) -> List[MultilayerInsight]:
        """
        Collect and aggregate insights from all system layers (Steps 19-26).
        
        Returns:
            List of multilayer insights
        """
        try:
            logger.info("Aggregating insights from all system layers...")
            
            insights = []
            
            # Layer 1: Collective Memory (Step 22)
            memory_insights = await self._collect_memory_insights()
            insights.extend(memory_insights)
            
            # Layer 2: Collective Intelligence (Step 23)
            intelligence_insights = await self._collect_intelligence_insights()
            insights.extend(intelligence_insights)
            
            # Layer 3: Meta-Optimization (Step 24)
            optimization_insights = await self._collect_optimization_insights()
            insights.extend(optimization_insights)
            
            # Layer 4: Adaptive Goals (Step 25)
            governance_insights = await self._collect_governance_insights()
            insights.extend(governance_insights)
            
            # Layer 5: Ethical Consensus (Step 26)
            consensus_insights = await self._collect_consensus_insights()
            insights.extend(consensus_insights)
            
            logger.info(f"Collected {len(insights)} insights from {5} layers")
            return insights
            
        except Exception as e:
            logger.error(f"Error aggregating multilayer insights: {e}")
            return []
    
    async def _collect_memory_insights(self) -> List[MultilayerInsight]:
        """Collect insights from collective memory system"""
        insights = []
        
        try:
            # Get recent distilled knowledge
            knowledge = await self.db.llm_distilled_knowledge.find().sort(
                "timestamp", -1
            ).limit(10).to_list(10)
            
            if knowledge:
                avg_confidence = np.mean([k.get("confidence_score", 0) for k in knowledge])
                knowledge_count = len(knowledge)
                
                insights.append(MultilayerInsight(
                    layer_name="collective_memory",
                    insight_type="knowledge_quality",
                    content=f"Memory system contains {knowledge_count} knowledge units with avg confidence {avg_confidence:.2f}",
                    confidence=avg_confidence,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    metrics={
                        "knowledge_count": float(knowledge_count),
                        "avg_confidence": avg_confidence,
                        "retention_quality": avg_confidence * 100
                    }
                ))
            else:
                # Sample data mode
                insights.append(self._generate_sample_memory_insight())
            
        except Exception as e:
            logger.error(f"Error collecting memory insights: {e}")
            insights.append(self._generate_sample_memory_insight())
        
        return insights
    
    async def _collect_intelligence_insights(self) -> List[MultilayerInsight]:
        """Collect insights from collective intelligence system"""
        insights = []
        
        try:
            # Get recent synthesis results
            synthesis = await self.db.llm_synthesis_results.find().sort(
                "timestamp", -1
            ).limit(10).to_list(10)
            
            if synthesis:
                avg_consensus = np.mean([s.get("consensus_score", 0) for s in synthesis])
                avg_confidence = np.mean([s.get("overall_confidence", 0) for s in synthesis])
                
                insights.append(MultilayerInsight(
                    layer_name="collective_intelligence",
                    insight_type="synthesis_quality",
                    content=f"Intelligence layer achieving {avg_consensus:.1%} consensus with {avg_confidence:.1%} confidence",
                    confidence=avg_confidence,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    metrics={
                        "consensus_score": avg_consensus,
                        "synthesis_confidence": avg_confidence,
                        "intelligence_index": (avg_consensus + avg_confidence) / 2 * 100
                    }
                ))
            else:
                insights.append(self._generate_sample_intelligence_insight())
            
        except Exception as e:
            logger.error(f"Error collecting intelligence insights: {e}")
            insights.append(self._generate_sample_intelligence_insight())
        
        return insights
    
    async def _collect_optimization_insights(self) -> List[MultilayerInsight]:
        """Collect insights from meta-optimization system"""
        insights = []
        
        try:
            # Get recent optimization cycles
            optimizations = await self.db.llm_meta_optimization_log.find().sort(
                "timestamp", -1
            ).limit(5).to_list(5)
            
            if optimizations:
                avg_health = np.mean([o.get("system_health_score", 75) for o in optimizations])
                applied_count = sum(1 for o in optimizations if o.get("applied", False))
                
                insights.append(MultilayerInsight(
                    layer_name="meta_optimization",
                    insight_type="system_health",
                    content=f"Optimization layer maintains {avg_health:.1f}% system health, {applied_count}/{len(optimizations)} cycles applied",
                    confidence=0.85,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    metrics={
                        "system_health": avg_health,
                        "optimization_efficacy": applied_count / len(optimizations) if optimizations else 0,
                        "adaptation_rate": float(applied_count)
                    }
                ))
            else:
                insights.append(self._generate_sample_optimization_insight())
            
        except Exception as e:
            logger.error(f"Error collecting optimization insights: {e}")
            insights.append(self._generate_sample_optimization_insight())
        
        return insights
    
    async def _collect_governance_insights(self) -> List[MultilayerInsight]:
        """Collect insights from adaptive goal/governance system"""
        insights = []
        
        try:
            # Get recent governance logs
            governance = await self.db.llm_governance_log.find().sort(
                "timestamp", -1
            ).limit(20).to_list(20)
            
            if governance:
                avg_alignment = np.mean([g.get("overall_alignment", 0) for g in governance])
                execution_rate = sum(1 for g in governance if g.get("executed", False)) / len(governance)
                
                insights.append(MultilayerInsight(
                    layer_name="adaptive_governance",
                    insight_type="goal_alignment",
                    content=f"Governance achieving {avg_alignment:.1%} alignment, {execution_rate:.1%} execution rate",
                    confidence=0.88,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    metrics={
                        "governance_alignment": avg_alignment * 100,
                        "execution_rate": execution_rate * 100,
                        "goal_efficacy": (avg_alignment * execution_rate) * 100
                    }
                ))
            else:
                insights.append(self._generate_sample_governance_insight())
            
        except Exception as e:
            logger.error(f"Error collecting governance insights: {e}")
            insights.append(self._generate_sample_governance_insight())
        
        return insights
    
    async def _collect_consensus_insights(self) -> List[MultilayerInsight]:
        """Collect insights from ethical consensus system"""
        insights = []
        
        try:
            # Get recent consensus logs
            consensus = await self.db.llm_ethics_consensus_log.find().sort(
                "timestamp", -1
            ).limit(10).to_list(10)
            
            if consensus:
                avg_eai = np.mean([c.get("agreement_score", 0) for c in consensus])
                avg_variance = np.mean([c.get("agreement_variance", 0) for c in consensus])
                consensus_rate = sum(1 for c in consensus if c.get("consensus_reached", False)) / len(consensus)
                
                insights.append(MultilayerInsight(
                    layer_name="ethical_consensus",
                    insight_type="consensus_quality",
                    content=f"Ethical consensus at {avg_eai:.2f} EAI, {avg_variance:.3f} variance, {consensus_rate:.1%} agreement rate",
                    confidence=0.90,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    metrics={
                        "ethical_alignment_index": avg_eai * 100,
                        "consensus_variance": avg_variance,
                        "consensus_rate": consensus_rate * 100,
                        "ethical_stability": (1 - avg_variance) * 100
                    }
                ))
            else:
                insights.append(self._generate_sample_consensus_insight())
            
        except Exception as e:
            logger.error(f"Error collecting consensus insights: {e}")
            insights.append(self._generate_sample_consensus_insight())
        
        return insights
    
    # Sample data generators for when real data is unavailable
    def _generate_sample_memory_insight(self) -> MultilayerInsight:
        """Generate sample memory insight [MOCK]"""
        return MultilayerInsight(
            layer_name="collective_memory",
            insight_type="knowledge_quality",
            content="Memory system contains 15 knowledge units with avg confidence 0.82 [SAMPLE DATA]",
            confidence=0.82,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metrics={
                "knowledge_count": 15.0,
                "avg_confidence": 0.82,
                "retention_quality": 82.0
            }
        )
    
    def _generate_sample_intelligence_insight(self) -> MultilayerInsight:
        """Generate sample intelligence insight [MOCK]"""
        return MultilayerInsight(
            layer_name="collective_intelligence",
            insight_type="synthesis_quality",
            content="Intelligence layer achieving 87% consensus with 84% confidence [SAMPLE DATA]",
            confidence=0.84,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metrics={
                "consensus_score": 0.87,
                "synthesis_confidence": 0.84,
                "intelligence_index": 85.5
            }
        )
    
    def _generate_sample_optimization_insight(self) -> MultilayerInsight:
        """Generate sample optimization insight [MOCK]"""
        return MultilayerInsight(
            layer_name="meta_optimization",
            insight_type="system_health",
            content="Optimization layer maintains 88.5% system health, 3/5 cycles applied [SAMPLE DATA]",
            confidence=0.85,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metrics={
                "system_health": 88.5,
                "optimization_efficacy": 0.60,
                "adaptation_rate": 3.0
            }
        )
    
    def _generate_sample_governance_insight(self) -> MultilayerInsight:
        """Generate sample governance insight [MOCK]"""
        return MultilayerInsight(
            layer_name="adaptive_governance",
            insight_type="goal_alignment",
            content="Governance achieving 86% alignment, 72% execution rate [SAMPLE DATA]",
            confidence=0.88,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metrics={
                "governance_alignment": 86.0,
                "execution_rate": 72.0,
                "goal_efficacy": 61.92
            }
        )
    
    def _generate_sample_consensus_insight(self) -> MultilayerInsight:
        """Generate sample consensus insight [MOCK]"""
        return MultilayerInsight(
            layer_name="ethical_consensus",
            insight_type="consensus_quality",
            content="Ethical consensus at 0.89 EAI, 0.08 variance, 92% agreement rate [SAMPLE DATA]",
            confidence=0.90,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metrics={
                "ethical_alignment_index": 89.0,
                "consensus_variance": 0.08,
                "consensus_rate": 92.0,
                "ethical_stability": 92.0
            }
        )
    
    async def derive_cognitive_patterns(
        self, 
        insights: List[MultilayerInsight]
    ) -> List[CognitivePattern]:
        """
        Identify meta-patterns and emergent strategies from cross-layer insights.
        
        Args:
            insights: Collected multilayer insights
        
        Returns:
            List of detected cognitive patterns
        """
        try:
            logger.info("Deriving cognitive patterns from insights...")
            
            patterns = []
            
            # Get historical patterns for trend analysis
            historical_patterns = await self.db.llm_cognitive_patterns.find().sort(
                "last_detected", -1
            ).limit(20).to_list(20)
            
            # Pattern 1: Trust-Memory Harmonization
            trust_memory_pattern = self._detect_trust_memory_pattern(insights, historical_patterns)
            if trust_memory_pattern:
                patterns.append(trust_memory_pattern)
            
            # Pattern 2: Predictive Alignment
            predictive_pattern = self._detect_predictive_alignment_pattern(insights, historical_patterns)
            if predictive_pattern:
                patterns.append(predictive_pattern)
            
            # Pattern 3: Ethical Convergence
            ethical_pattern = self._detect_ethical_convergence_pattern(insights, historical_patterns)
            if ethical_pattern:
                patterns.append(ethical_pattern)
            
            # Pattern 4: Adaptive Stability
            stability_pattern = self._detect_adaptive_stability_pattern(insights, historical_patterns)
            if stability_pattern:
                patterns.append(stability_pattern)
            
            logger.info(f"Detected {len(patterns)} cognitive patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Error deriving cognitive patterns: {e}")
            return []
    
    def _detect_trust_memory_pattern(
        self, 
        insights: List[MultilayerInsight],
        historical: List[Dict]
    ) -> Optional[CognitivePattern]:
        """Detect trust-memory harmonization pattern"""
        try:
            # Find relevant insights
            memory_insights = [i for i in insights if i.layer_name == "collective_memory"]
            consensus_insights = [i for i in insights if i.layer_name == "ethical_consensus"]
            
            if not memory_insights or not consensus_insights:
                return None
            
            # Calculate pattern strength
            memory_quality = memory_insights[0].metrics.get("retention_quality", 0) / 100
            consensus_stability = consensus_insights[0].metrics.get("ethical_stability", 0) / 100
            
            strength = (memory_quality * 0.5 + consensus_stability * 0.5)
            
            if strength < self.pattern_strength_threshold:
                return None
            
            # Check historical occurrences
            historical_count = sum(1 for p in historical if p.get("pattern_name") == "Trust-Memory Harmonization")
            
            if historical_count == 0:
                first_detected = datetime.now(timezone.utc).isoformat()
            else:
                first_detected = min([p.get("first_detected", datetime.now(timezone.utc).isoformat()) 
                                     for p in historical if p.get("pattern_name") == "Trust-Memory Harmonization"])
            
            return CognitivePattern(
                pattern_id=str(uuid.uuid4()),
                pattern_name="Trust-Memory Harmonization",
                description="System demonstrates aligned trust calibration and memory retention, enabling predictive consensus",
                layers_involved=["collective_memory", "ethical_consensus"],
                strength=strength,
                emergence_count=historical_count + 1,
                first_detected=first_detected,
                last_detected=datetime.now(timezone.utc).isoformat(),
                impact_areas=["consensus_stability", "decision_quality", "value_preservation"]
            )
            
        except Exception as e:
            logger.error(f"Error detecting trust-memory pattern: {e}")
            return None
    
    def _detect_predictive_alignment_pattern(
        self,
        insights: List[MultilayerInsight],
        historical: List[Dict]
    ) -> Optional[CognitivePattern]:
        """Detect predictive alignment pattern"""
        try:
            intelligence_insights = [i for i in insights if i.layer_name == "collective_intelligence"]
            governance_insights = [i for i in insights if i.layer_name == "adaptive_governance"]
            
            if not intelligence_insights or not governance_insights:
                return None
            
            intelligence_index = intelligence_insights[0].metrics.get("intelligence_index", 0) / 100
            governance_alignment = governance_insights[0].metrics.get("governance_alignment", 0) / 100
            
            strength = (intelligence_index * 0.6 + governance_alignment * 0.4)
            
            if strength < self.pattern_strength_threshold:
                return None
            
            historical_count = sum(1 for p in historical if p.get("pattern_name") == "Predictive Alignment")
            
            return CognitivePattern(
                pattern_id=str(uuid.uuid4()),
                pattern_name="Predictive Alignment",
                description="Intelligence synthesis predicts governance outcomes, enabling proactive value alignment",
                layers_involved=["collective_intelligence", "adaptive_governance"],
                strength=strength,
                emergence_count=historical_count + 1,
                first_detected=min([p.get("first_detected", datetime.now(timezone.utc).isoformat()) 
                                   for p in historical if p.get("pattern_name") == "Predictive Alignment"]) 
                                   if historical_count > 0 else datetime.now(timezone.utc).isoformat(),
                last_detected=datetime.now(timezone.utc).isoformat(),
                impact_areas=["goal_formation", "strategic_planning", "alignment_accuracy"]
            )
            
        except Exception as e:
            logger.error(f"Error detecting predictive alignment pattern: {e}")
            return None
    
    def _detect_ethical_convergence_pattern(
        self,
        insights: List[MultilayerInsight],
        historical: List[Dict]
    ) -> Optional[CognitivePattern]:
        """Detect ethical convergence pattern"""
        try:
            consensus_insights = [i for i in insights if i.layer_name == "ethical_consensus"]
            governance_insights = [i for i in insights if i.layer_name == "adaptive_governance"]
            
            if not consensus_insights or not governance_insights:
                return None
            
            eai = consensus_insights[0].metrics.get("ethical_alignment_index", 0) / 100
            gov_alignment = governance_insights[0].metrics.get("governance_alignment", 0) / 100
            
            strength = (eai * 0.6 + gov_alignment * 0.4)
            
            if strength < self.pattern_strength_threshold:
                return None
            
            historical_count = sum(1 for p in historical if p.get("pattern_name") == "Ethical Convergence")
            
            return CognitivePattern(
                pattern_id=str(uuid.uuid4()),
                pattern_name="Ethical Convergence",
                description="Consensus and governance layers converge on aligned ethical decisions with minimal conflict",
                layers_involved=["ethical_consensus", "adaptive_governance"],
                strength=strength,
                emergence_count=historical_count + 1,
                first_detected=min([p.get("first_detected", datetime.now(timezone.utc).isoformat()) 
                                   for p in historical if p.get("pattern_name") == "Ethical Convergence"]) 
                                   if historical_count > 0 else datetime.now(timezone.utc).isoformat(),
                last_detected=datetime.now(timezone.utc).isoformat(),
                impact_areas=["ethical_stability", "conflict_resolution", "value_integrity"]
            )
            
        except Exception as e:
            logger.error(f"Error detecting ethical convergence pattern: {e}")
            return None
    
    def _detect_adaptive_stability_pattern(
        self,
        insights: List[MultilayerInsight],
        historical: List[Dict]
    ) -> Optional[CognitivePattern]:
        """Detect adaptive stability pattern"""
        try:
            optimization_insights = [i for i in insights if i.layer_name == "meta_optimization"]
            
            if not optimization_insights:
                return None
            
            system_health = optimization_insights[0].metrics.get("system_health", 0) / 100
            
            if system_health < self.pattern_strength_threshold:
                return None
            
            historical_count = sum(1 for p in historical if p.get("pattern_name") == "Adaptive Stability")
            
            return CognitivePattern(
                pattern_id=str(uuid.uuid4()),
                pattern_name="Adaptive Stability",
                description="Meta-optimization maintains system stability while enabling continuous adaptation",
                layers_involved=["meta_optimization"],
                strength=system_health,
                emergence_count=historical_count + 1,
                first_detected=min([p.get("first_detected", datetime.now(timezone.utc).isoformat()) 
                                   for p in historical if p.get("pattern_name") == "Adaptive Stability"]) 
                                   if historical_count > 0 else datetime.now(timezone.utc).isoformat(),
                last_detected=datetime.now(timezone.utc).isoformat(),
                impact_areas=["system_stability", "adaptation_rate", "performance_consistency"]
            )
            
        except Exception as e:
            logger.error(f"Error detecting adaptive stability pattern: {e}")
            return None
    
    async def preserve_value_alignment(
        self, 
        insights: List[MultilayerInsight]
    ) -> Tuple[List[ValueState], Dict[str, Any]]:
        """
        Ensure long-term adherence to ethical and performance values.
        Track value drift and generate drift report.
        
        Args:
            insights: Current multilayer insights
        
        Returns:
            Tuple of (value_states, drift_report)
        """
        try:
            logger.info("Preserving value alignment and tracking drift...")
            
            value_states = []
            
            # Initialize baselines if needed
            await self._initialize_value_baselines()
            
            # Evaluate each core value
            for value_name, config in self.core_values.items():
                value_state = await self._evaluate_value_state(value_name, config, insights)
                value_states.append(value_state)
            
            # Generate drift report
            drift_report = self._generate_drift_report(value_states)
            
            # Store value states in database
            for state in value_states:
                await self.db.llm_value_preservation.insert_one(state.to_dict())
            
            logger.info(f"Preserved {len(value_states)} value states, drift: {drift_report['overall_drift']:.2%}")
            return value_states, drift_report
            
        except Exception as e:
            logger.error(f"Error preserving value alignment: {e}")
            return [], {}
    
    async def _initialize_value_baselines(self):
        """Initialize value baselines from current system state"""
        if self.value_baselines:
            return  # Already initialized
        
        try:
            # Get historical baselines or use defaults
            for value_name, config in self.core_values.items():
                baseline_doc = await self.db.llm_value_preservation.find_one(
                    {"value_name": value_name},
                    sort=[("last_updated", -1)]
                )
                
                if baseline_doc:
                    self.value_baselines[value_name] = baseline_doc.get("current_score", config["target"])
                else:
                    self.value_baselines[value_name] = config["target"]
            
            logger.info("Value baselines initialized")
            
        except Exception as e:
            logger.error(f"Error initializing value baselines: {e}")
            # Use targets as baselines
            for value_name, config in self.core_values.items():
                self.value_baselines[value_name] = config["target"]
    
    async def _evaluate_value_state(
        self,
        value_name: str,
        config: Dict,
        insights: List[MultilayerInsight]
    ) -> ValueState:
        """Evaluate current state of a specific value"""
        try:
            # Extract relevant metrics from insights
            current_score = await self._calculate_value_score(value_name, insights)
            
            # Get baseline
            baseline = self.value_baselines.get(value_name, config["target"])
            
            # Calculate drift
            drift_amount = current_score - baseline
            drift_pct = (drift_amount / baseline) if baseline > 0 else 0
            
            # Determine drift direction
            if abs(drift_pct) < 0.03:  # Less than 3% change
                drift_direction = "stable"
            elif drift_pct > 0:
                drift_direction = "positive"
            else:
                drift_direction = "negative"
            
            # Calculate stability index
            stability_index = max(0.0, 1.0 - abs(drift_pct))
            
            return ValueState(
                value_name=value_name,
                category=config["category"],
                current_score=current_score,
                target_score=config["target"],
                drift_amount=drift_amount,
                drift_direction=drift_direction,
                stability_index=stability_index,
                last_updated=datetime.now(timezone.utc).isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error evaluating value state for {value_name}: {e}")
            return ValueState(
                value_name=value_name,
                category=config["category"],
                current_score=config["target"],
                target_score=config["target"],
                drift_amount=0.0,
                drift_direction="stable",
                stability_index=1.0,
                last_updated=datetime.now(timezone.utc).isoformat()
            )
    
    async def _calculate_value_score(
        self, 
        value_name: str, 
        insights: List[MultilayerInsight]
    ) -> float:
        """Calculate current score for a specific value from insights"""
        
        # Map values to relevant metrics
        metric_mappings = {
            "transparency": ["governance_alignment", "retention_quality"],
            "fairness": ["ethical_alignment_index", "consensus_rate"],
            "safety": ["system_health", "ethical_stability"],
            "performance": ["intelligence_index", "goal_efficacy"],
            "alignment": ["governance_alignment", "ethical_alignment_index"],
            "stability": ["ethical_stability", "system_health"]
        }
        
        relevant_metrics = metric_mappings.get(value_name, [])
        
        # Extract metric values from insights
        metric_values = []
        for insight in insights:
            for metric_name in relevant_metrics:
                if metric_name in insight.metrics:
                    metric_values.append(insight.metrics[metric_name])
        
        # Calculate average score
        if metric_values:
            return float(np.mean(metric_values))
        else:
            # Default to target if no data
            return self.core_values[value_name]["target"]
    
    def _generate_drift_report(self, value_states: List[ValueState]) -> Dict[str, Any]:
        """Generate comprehensive drift report"""
        
        drift_amounts = [vs.drift_amount for vs in value_states]
        drift_percentages = [(vs.drift_amount / vs.target_score) if vs.target_score > 0 else 0 
                            for vs in value_states]
        
        overall_drift = np.mean([abs(d) for d in drift_percentages])
        
        # Categorize by drift direction
        positive_drift = [vs.value_name for vs in value_states if vs.drift_direction == "positive"]
        negative_drift = [vs.value_name for vs in value_states if vs.drift_direction == "negative"]
        stable = [vs.value_name for vs in value_states if vs.drift_direction == "stable"]
        
        # Calculate overall stability
        avg_stability = np.mean([vs.stability_index for vs in value_states])
        
        # Identify concerning drifts
        concerning_drifts = [
            f"{vs.value_name}: {vs.drift_amount:+.1f} ({vs.drift_amount/vs.target_score:+.1%})"
            for vs in value_states if abs(vs.drift_amount / vs.target_score) > 0.10
        ]
        
        return {
            "overall_drift": overall_drift,
            "avg_stability": avg_stability,
            "positive_drift_values": positive_drift,
            "negative_drift_values": negative_drift,
            "stable_values": stable,
            "concerning_drifts": concerning_drifts,
            "drift_status": "critical" if overall_drift > 0.15 else "moderate" if overall_drift > 0.08 else "healthy"
        }
    
    async def run_self_synthesis_cycle(self, trigger: str = "scheduled") -> SynthesisCycle:
        """
        Execute complete autonomous synthesis cycle.
        
        Args:
            trigger: What triggered the cycle ("scheduled", "manual", "threshold")
        
        Returns:
            SynthesisCycle with complete results and recommendations
        """
        cycle_id = str(uuid.uuid4())
        logger.info(f"Starting synthesis cycle {cycle_id} (trigger: {trigger})")
        
        try:
            # Step 1: Aggregate multilayer insights
            insights = await self.aggregate_multilayer_insights()
            
            # Step 2: Derive cognitive patterns
            patterns = await self.derive_cognitive_patterns(insights)
            
            # Step 3: Preserve value alignment
            value_states, drift_report = await self.preserve_value_alignment(insights)
            
            # Step 4: Calculate stability metrics
            stability_metrics = self._calculate_stability_metrics(insights, patterns, drift_report)
            
            # Step 5: Calculate cognitive coherence index
            cognitive_coherence = self._calculate_cognitive_coherence(insights, patterns)
            
            # Step 6: Calculate value integrity score
            value_integrity = self._calculate_value_integrity(value_states, drift_report)
            
            # Step 7: Generate reflection summary
            reflection = await self._generate_reflection_summary(
                insights, patterns, value_states, drift_report, cognitive_coherence, value_integrity
            )
            
            # Step 8: Generate recommendations
            recommendations = self._generate_synthesis_recommendations(
                patterns, value_states, drift_report, cognitive_coherence, value_integrity
            )
            
            # Create synthesis cycle result
            cycle = SynthesisCycle(
                cycle_id=cycle_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                trigger=trigger,
                layers_integrated=list(set([i.layer_name for i in insights])),
                insights_collected=len(insights),
                patterns_detected=patterns,
                value_states=value_states,
                drift_report=drift_report,
                stability_metrics=stability_metrics,
                reflection_summary=reflection,
                cognitive_coherence_index=cognitive_coherence,
                value_integrity_score=value_integrity,
                recommendations=recommendations
            )
            
            # Store in database
            await self.db.llm_cognitive_synthesis_log.insert_one(cycle.to_dict())
            
            # Update pattern history
            for pattern in patterns:
                await self.db.llm_cognitive_patterns.update_one(
                    {"pattern_name": pattern.pattern_name},
                    {"$set": pattern.to_dict()},
                    upsert=True
                )
            
            logger.info(
                f"Synthesis cycle {cycle_id} complete: "
                f"Coherence={cognitive_coherence:.2f}, Integrity={value_integrity:.1f}%, "
                f"Patterns={len(patterns)}, Drift={drift_report.get('drift_status', 'unknown')}"
            )
            
            return cycle
            
        except Exception as e:
            logger.error(f"Error in synthesis cycle: {e}")
            return SynthesisCycle(
                cycle_id=cycle_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                trigger=trigger,
                layers_integrated=[],
                insights_collected=0,
                patterns_detected=[],
                value_states=[],
                drift_report={"error": str(e)},
                stability_metrics={},
                reflection_summary=f"Cycle failed: {str(e)}",
                cognitive_coherence_index=0.0,
                value_integrity_score=50.0,
                recommendations=["System error - review logs"]
            )
    
    def _calculate_stability_metrics(
        self,
        insights: List[MultilayerInsight],
        patterns: List[CognitivePattern],
        drift_report: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate system stability metrics"""
        
        # Insight confidence stability
        insight_confidence = np.mean([i.confidence for i in insights]) if insights else 0.5
        
        # Pattern strength stability
        pattern_strength = np.mean([p.strength for p in patterns]) if patterns else 0.5
        
        # Value drift stability
        drift_stability = 1.0 - drift_report.get("overall_drift", 0.5)
        
        # Overall stability index
        overall_stability = (insight_confidence * 0.3 + pattern_strength * 0.3 + drift_stability * 0.4)
        
        return {
            "insight_confidence_stability": round(insight_confidence, 3),
            "pattern_strength_stability": round(pattern_strength, 3),
            "value_drift_stability": round(drift_stability, 3),
            "overall_stability_index": round(overall_stability, 3)
        }
    
    def _calculate_cognitive_coherence(
        self,
        insights: List[MultilayerInsight],
        patterns: List[CognitivePattern]
    ) -> float:
        """Calculate Cognitive Coherence Index (0-1)"""
        
        # Component 1: Insight quality (avg confidence)
        insight_quality = np.mean([i.confidence for i in insights]) if insights else 0.5
        
        # Component 2: Pattern emergence (normalized)
        pattern_score = min(1.0, len(patterns) / 4.0) if patterns else 0.0
        
        # Component 3: Cross-layer integration (unique layers / total possible)
        unique_layers = len(set([i.layer_name for i in insights]))
        integration_score = unique_layers / 5.0  # 5 total layers
        
        # Weighted coherence
        coherence = (insight_quality * 0.4 + pattern_score * 0.3 + integration_score * 0.3)
        
        return round(min(1.0, max(0.0, coherence)), 3)
    
    def _calculate_value_integrity(
        self,
        value_states: List[ValueState],
        drift_report: Dict[str, Any]
    ) -> float:
        """Calculate Value Integrity Score (0-100)"""
        
        if not value_states:
            return 50.0
        
        # Component 1: Average value stability
        avg_stability = np.mean([vs.stability_index for vs in value_states])
        
        # Component 2: Drift severity
        overall_drift = drift_report.get("overall_drift", 0.10)
        drift_score = max(0.0, 1.0 - (overall_drift * 5))  # Penalize drift
        
        # Component 3: Values at or above target
        values_meeting_target = sum(1 for vs in value_states if vs.current_score >= vs.target_score * 0.95)
        target_score = values_meeting_target / len(value_states)
        
        # Weighted integrity score
        integrity = (avg_stability * 0.4 + drift_score * 0.3 + target_score * 0.3) * 100
        
        return round(min(100.0, max(0.0, integrity)), 1)
    
    async def _generate_reflection_summary(
        self,
        insights: List[MultilayerInsight],
        patterns: List[CognitivePattern],
        value_states: List[ValueState],
        drift_report: Dict[str, Any],
        coherence: float,
        integrity: float
    ) -> str:
        """Generate reflective summary using LLM if available"""
        
        try:
            if self.llm_available:
                return await self._llm_generate_reflection(
                    insights, patterns, value_states, drift_report, coherence, integrity
                )
            else:
                return self._fallback_reflection(
                    insights, patterns, value_states, drift_report, coherence, integrity
                )
        except Exception as e:
            logger.error(f"Error generating reflection: {e}")
            return self._fallback_reflection(
                insights, patterns, value_states, drift_report, coherence, integrity
            )
    
    async def _llm_generate_reflection(
        self,
        insights: List[MultilayerInsight],
        patterns: List[CognitivePattern],
        value_states: List[ValueState],
        drift_report: Dict[str, Any],
        coherence: float,
        integrity: float
    ) -> str:
        """Use LLM to generate reflective summary [PROD]"""
        
        try:
            chat = LlmChat(
                api_key=self.llm_key,
                session_id=f"synthesis-reflection-{uuid.uuid4()}",
                system_message="You are an AI meta-cognition analyzer for AlphaZero Chess. Generate concise, insightful reflections on system synthesis cycles."
            ).with_model("openai", "gpt-4o-mini")
            
            # Build comprehensive prompt
            patterns_summary = "\n".join([f"- {p.pattern_name}: {p.description} (strength: {p.strength:.2f})" 
                                         for p in patterns])
            
            value_summary = "\n".join([f"- {vs.value_name}: {vs.current_score:.1f} (target: {vs.target_score:.1f}, drift: {vs.drift_direction})"
                                      for vs in value_states])
            
            prompt = f"""Analyze the following cognitive synthesis cycle and generate a 3-4 sentence reflective summary:

**Synthesis Metrics:**
- Cognitive Coherence Index: {coherence:.2f}
- Value Integrity Score: {integrity:.1f}%
- Insights Collected: {len(insights)} from {len(set([i.layer_name for i in insights]))} layers
- Patterns Detected: {len(patterns)}

**Detected Patterns:**
{patterns_summary if patterns else "No significant patterns detected"}

**Value States:**
{value_summary}

**Value Drift Status:** {drift_report.get('drift_status', 'unknown')}
**Overall Drift:** {drift_report.get('overall_drift', 0):.1%}

Generate a concise reflection that:
1. Summarizes the key cognitive achievements
2. Highlights any emergent meta-patterns
3. Notes value integrity status
4. Provides one forward-looking insight

Keep it professional, clear, and under 100 words."""
            
            user_message = UserMessage(text=prompt)
            response = await chat.send_message(user_message)
            
            reflection = response.strip()
            logger.info("[PROD] LLM-generated reflection created")
            return reflection
            
        except Exception as e:
            logger.error(f"Error in LLM reflection generation: {e}")
            return self._fallback_reflection(insights, patterns, value_states, drift_report, coherence, integrity)
    
    def _fallback_reflection(
        self,
        insights: List[MultilayerInsight],
        patterns: List[CognitivePattern],
        value_states: List[ValueState],
        drift_report: Dict[str, Any],
        coherence: float,
        integrity: float
    ) -> str:
        """Fallback reflection generation [MOCK]"""
        
        reflection_parts = []
        
        # Synthesis overview
        reflection_parts.append(
            f"Synthesis Cycle complete: {len(insights)} insights integrated from {len(set([i.layer_name for i in insights]))} system layers."
        )
        
        # Pattern detection
        if patterns:
            top_pattern = max(patterns, key=lambda p: p.strength)
            reflection_parts.append(
                f"New meta-pattern detected: '{top_pattern.pattern_name}' with {top_pattern.strength:.1%} strength."
            )
        else:
            reflection_parts.append("No new significant patterns emerged this cycle.")
        
        # Value integrity
        drift_status = drift_report.get("drift_status", "moderate")
        reflection_parts.append(
            f"Value Drift: {drift_report.get('overall_drift', 0):.1%} ({drift_status}). Integrity Score: {integrity:.1f}%."
        )
        
        # Coherence status
        coherence_status = "High" if coherence >= 0.80 else "Moderate" if coherence >= 0.65 else "Low"
        reflection_parts.append(
            f"Cognitive Coherence: {coherence_status} ({coherence:.2f}). System demonstrating {coherence_status.lower()} meta-adaptive capability."
        )
        
        return " ".join(reflection_parts) + " [SAMPLE REFLECTION]"
    
    def _generate_synthesis_recommendations(
        self,
        patterns: List[CognitivePattern],
        value_states: List[ValueState],
        drift_report: Dict[str, Any],
        coherence: float,
        integrity: float
    ) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Coherence recommendations
        if coherence < 0.70:
            recommendations.append("⚠️ Low cognitive coherence detected. Review layer integration quality.")
        elif coherence >= 0.85:
            recommendations.append("✅ Excellent cognitive coherence. System demonstrating strong meta-adaptation.")
        
        # Integrity recommendations
        if integrity < 75.0:
            recommendations.append("⚠️ Value integrity below threshold. Implement value reinforcement measures.")
        elif integrity >= 90.0:
            recommendations.append("✅ Outstanding value integrity. Core values well-preserved.")
        
        # Drift recommendations
        drift_status = drift_report.get("drift_status", "moderate")
        if drift_status == "critical":
            recommendations.append("🚨 Critical value drift detected. Immediate intervention required.")
        elif drift_status == "healthy":
            recommendations.append("✅ Value drift within healthy bounds. Continue monitoring.")
        
        # Pattern recommendations
        if len(patterns) >= 3:
            recommendations.append(f"📊 {len(patterns)} cognitive patterns detected. System showing emergent intelligence.")
        elif len(patterns) == 0:
            recommendations.append("⚠️ No patterns detected. May indicate low cross-layer interaction.")
        
        # Concerning drifts
        concerning = drift_report.get("concerning_drifts", [])
        if concerning:
            recommendations.append(f"⚠️ Concerning drift in: {', '.join([c.split(':')[0] for c in concerning[:2]])}")
        
        # Default recommendation
        if not recommendations:
            recommendations.append("✅ System operating within optimal parameters. Continue scheduled synthesis cycles.")
        
        return recommendations
    
    async def get_synthesis_status(self) -> Dict[str, Any]:
        """Get current synthesis system status"""
        try:
            # Get latest cycle
            latest_cycle = await self.db.llm_cognitive_synthesis_log.find_one(
                sort=[("timestamp", -1)]
            )
            
            # Get recent patterns
            recent_patterns = await self.db.llm_cognitive_patterns.find().sort(
                "last_detected", -1
            ).limit(5).to_list(5)
            
            # Get recent value states
            recent_values = await self.db.llm_value_preservation.find().sort(
                "last_updated", -1
            ).limit(6).to_list(6)  # One per core value
            
            if latest_cycle:
                status = {
                    "last_synthesis": latest_cycle.get("timestamp"),
                    "cognitive_coherence_index": latest_cycle.get("cognitive_coherence_index", 0),
                    "value_integrity_score": latest_cycle.get("value_integrity_score", 0),
                    "active_patterns": len(recent_patterns),
                    "drift_status": latest_cycle.get("drift_report", {}).get("drift_status", "unknown"),
                    "system_status": "operational",
                    "latest_patterns": [p.get("pattern_name") for p in recent_patterns[:3]],
                    "value_summary": {
                        v.get("value_name"): {
                            "score": v.get("current_score"),
                            "drift": v.get("drift_direction")
                        }
                        for v in recent_values
                    }
                }
            else:
                status = {
                    "last_synthesis": None,
                    "cognitive_coherence_index": 0.0,
                    "value_integrity_score": 0.0,
                    "active_patterns": 0,
                    "drift_status": "unknown",
                    "system_status": "initializing",
                    "latest_patterns": [],
                    "value_summary": {}
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting synthesis status: {e}")
            return {
                "system_status": "error",
                "error": str(e)
            }
