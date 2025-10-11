"""
Cognitive Resonance & Long-Term System Stability (Step 34)

This module implements the Cognitive Resonance Framework (CRF) that ensures stable,
long-term synchronization of the system's creative, reflective, mnemonic, cohesive,
and ethical layers.

Core Design Principles:
1. Resonance Matrix Engine (RME) - Models cross-module influence patterns
2. Temporal Stability Monitor (TSM) - Tracks stability across extended play sessions
3. Adaptive Feedback Regulator (AFR) - Balances learning rates dynamically
4. Entropy Control Mechanism (ECM) - Limits runaway novelty or stagnation
5. Persistence Resonator (PR) - Integrates long-term memory with ethical boundaries

Target Metrics:
- Resonance Index: ≥ 0.90
- Temporal Stability: ≥ 0.88
- Feedback Equilibrium: 0.50 ± 0.05
- Entropy Balance: 0.45–0.55
"""

import logging
import asyncio
import uuid
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
import os

# Import emergentintegrations for multi-provider LLM support
from emergentintegrations.llm.chat import LlmChat, UserMessage

logger = logging.getLogger(__name__)


@dataclass
class ResonanceMetrics:
    """Comprehensive resonance and stability metrics"""
    timestamp: str
    resonance_index: float  # 0-1: Overall system resonance (target ≥ 0.90)
    temporal_stability: float  # 0-1: Stability across time (target ≥ 0.88)
    feedback_equilibrium: float  # 0-1: Balance of feedback weights (target 0.50 ± 0.05)
    entropy_balance: float  # 0-1: Novelty vs stability balance (target 0.45-0.55)
    
    # Cross-module alignment scores
    creativity_reflection_alignment: float
    reflection_memory_alignment: float
    memory_cohesion_alignment: float
    cohesion_ethics_alignment: float
    ethics_creativity_alignment: float
    
    # Stability indicators
    drift_velocity: float  # Rate of parameter change
    oscillation_amplitude: float  # Degree of parameter fluctuation
    convergence_score: float  # How close system is to equilibrium
    
    # Module-specific resonance
    creativity_resonance: float
    reflection_resonance: float
    memory_resonance: float
    cohesion_resonance: float
    ethics_resonance: float
    
    # System health
    resonance_health: str  # "excellent", "good", "moderate", "needs_attention"
    stability_warnings: List[str]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class StabilityForecast:
    """Predicted stability trends for upcoming sessions"""
    forecast_id: str
    timestamp: str
    forecast_horizon_hours: int  # How far ahead this forecast predicts
    
    # Predicted metrics
    predicted_resonance_index: float
    predicted_temporal_stability: float
    predicted_entropy_balance: float
    
    # Trend predictions
    resonance_trend: str  # "improving", "stable", "declining"
    stability_trend: str
    entropy_trend: str
    
    # Risk assessment
    drift_risk_level: str  # "low", "medium", "high", "critical"
    oscillation_risk_level: str
    stagnation_risk_level: str
    
    # Confidence in predictions
    confidence_score: float  # 0-1
    
    # Recommended actions
    recommended_interventions: List[str]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ResonanceSnapshot:
    """Historical snapshot of system resonance state"""
    snapshot_id: str
    timestamp: str
    
    # Module states at snapshot time
    module_states: Dict[str, Dict[str, float]]
    
    # Resonance metrics
    resonance_metrics: Dict[str, float]
    
    # Session context
    game_sessions_count: int
    ethics_scans_count: int
    cohesion_cycles_count: int
    
    # Notable events
    events: List[str]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ResonanceReport:
    """Comprehensive resonance report (Report #001)"""
    report_id: str
    report_number: int
    timestamp: str
    reporting_period: Dict[str, str]  # start, end
    
    # Summary metrics
    overall_resonance_index: float
    overall_temporal_stability: float
    overall_feedback_equilibrium: float
    overall_entropy_balance: float
    
    # Module analysis
    module_resonance_scores: Dict[str, float]
    module_alignment_matrix: Dict[str, Dict[str, float]]
    
    # Stability analysis
    drift_analysis: Dict[str, Any]
    oscillation_analysis: Dict[str, Any]
    entropy_analysis: Dict[str, Any]
    
    # Forecasts
    short_term_forecast: Dict[str, Any]
    long_term_forecast: Dict[str, Any]
    
    # Trends
    resonance_trajectory: List[float]
    stability_trajectory: List[float]
    entropy_trajectory: List[float]
    
    # Recommendations
    recommendations: List[str]
    intervention_priorities: List[Dict[str, Any]]
    
    # Health assessment
    system_health_summary: str
    target_compliance: Dict[str, bool]
    
    def to_dict(self):
        return asdict(self)


class CognitiveResonanceController:
    """
    Cognitive Resonance & Long-Term System Stability Controller
    
    Establishes stable, long-term synchronization across all cognitive subsystems
    (Creativity, Reflection, Memory, Cohesion, Ethics) to prevent oscillations,
    over-adaptation, or drift while preserving adaptive flexibility.
    """
    
    def __init__(self, db_client):
        self.db = db_client
        self.llm_key = os.environ.get('EMERGENT_LLM_KEY')
        
        if not self.llm_key:
            logger.warning("EMERGENT_LLM_KEY not found - resonance will use mock mode")
            self.llm_available = False
        else:
            self.llm_available = True
            logger.info("Cognitive Resonance Controller initialized with multi-provider LLM support")
        
        # Target metrics (from problem statement)
        self.target_metrics = {
            "resonance_index": 0.90,
            "temporal_stability": 0.88,
            "feedback_equilibrium": 0.50,
            "feedback_equilibrium_tolerance": 0.05,
            "entropy_balance_min": 0.45,
            "entropy_balance_max": 0.55
        }
        
        # Stability thresholds
        self.stability_thresholds = {
            "max_drift_velocity": 0.05,  # Maximum rate of parameter change per cycle
            "max_oscillation_amplitude": 0.10,  # Maximum fluctuation range
            "min_convergence_score": 0.85  # Minimum convergence to equilibrium
        }
        
        # Entropy control parameters
        self.entropy_params = {
            "min_novelty": 0.40,  # Prevent stagnation
            "max_novelty": 0.80,  # Prevent runaway creativity
            "balance_point": 0.50  # Optimal balance
        }
        
        # LLM providers (per problem statement)
        self.llm_providers = {
            "primary": {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
            "secondary": {"provider": "openai", "model": "gpt-4o-mini"},
            "fallback": {"provider": "google", "model": "gemini-2.0-flash-exp"}
        }
        
        # Report counter
        self.report_counter = 0
        
        logger.info(
            f"Cognitive Resonance Framework initialized: "
            f"Resonance Target={self.target_metrics['resonance_index']}, "
            f"Stability Target={self.target_metrics['temporal_stability']}"
        )
    
    async def analyze_resonance_state(self) -> ResonanceMetrics:
        """
        Resonance Matrix Engine (RME)
        
        Correlate metrics from all active modules (Steps 29-33) to compute
        cross-module influence patterns and alignment forces.
        
        Returns:
            Comprehensive resonance metrics
        """
        try:
            logger.info("RME: Analyzing resonance state across all modules...")
            
            # Gather module states
            module_states = await self._gather_module_states()
            
            # Calculate cross-module alignments
            alignments = await self._calculate_cross_module_alignments(module_states)
            
            # Calculate stability indicators
            drift_velocity = await self._calculate_drift_velocity()
            oscillation_amplitude = await self._calculate_oscillation_amplitude()
            convergence_score = await self._calculate_convergence_score(module_states)
            
            # Calculate module-specific resonance
            module_resonance = await self._calculate_module_resonance(module_states)
            
            # Calculate overall resonance index
            resonance_index = await self._calculate_resonance_index(
                alignments, module_resonance, convergence_score
            )
            
            # Calculate temporal stability
            temporal_stability = await self._calculate_temporal_stability(
                drift_velocity, oscillation_amplitude
            )
            
            # Calculate feedback equilibrium
            feedback_equilibrium = await self._calculate_feedback_equilibrium(module_states)
            
            # Calculate entropy balance
            entropy_balance = await self._calculate_entropy_balance(module_states)
            
            # Determine health status
            resonance_health, warnings = self._determine_resonance_health(
                resonance_index, temporal_stability, feedback_equilibrium, entropy_balance
            )
            
            metrics = ResonanceMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                resonance_index=round(resonance_index, 3),
                temporal_stability=round(temporal_stability, 3),
                feedback_equilibrium=round(feedback_equilibrium, 3),
                entropy_balance=round(entropy_balance, 3),
                creativity_reflection_alignment=round(alignments["creativity_reflection"], 3),
                reflection_memory_alignment=round(alignments["reflection_memory"], 3),
                memory_cohesion_alignment=round(alignments["memory_cohesion"], 3),
                cohesion_ethics_alignment=round(alignments["cohesion_ethics"], 3),
                ethics_creativity_alignment=round(alignments["ethics_creativity"], 3),
                drift_velocity=round(drift_velocity, 3),
                oscillation_amplitude=round(oscillation_amplitude, 3),
                convergence_score=round(convergence_score, 3),
                creativity_resonance=round(module_resonance["creativity"], 3),
                reflection_resonance=round(module_resonance["reflection"], 3),
                memory_resonance=round(module_resonance["memory"], 3),
                cohesion_resonance=round(module_resonance["cohesion"], 3),
                ethics_resonance=round(module_resonance["ethics"], 3),
                resonance_health=resonance_health,
                stability_warnings=warnings
            )
            
            # Store metrics
            await self.db.llm_resonance_metrics.insert_one(metrics.to_dict())
            
            logger.info(
                f"Resonance state analyzed: Index={resonance_index:.3f}, "
                f"Stability={temporal_stability:.3f}, Health={resonance_health}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing resonance state: {e}")
            raise
    
    async def _gather_module_states(self) -> Dict[str, Dict[str, Any]]:
        """Gather current state from all modules"""
        try:
            states = {}
            
            # Creativity (Step 29)
            try:
                creativity_metrics = await self.db.llm_creative_synthesis.find().sort(
                    "timestamp", -1
                ).limit(10).to_list(10)
                if creativity_metrics:
                    states["creativity"] = {
                        "avg_novelty": np.mean([s.get("novelty_score", 0.7) for s in creativity_metrics]),
                        "avg_stability": np.mean([s.get("stability_score", 0.65) for s in creativity_metrics]),
                        "avg_ethical": np.mean([s.get("ethical_alignment", 0.9) for s in creativity_metrics]),
                        "count": len(creativity_metrics),
                        "health": len([s for s in creativity_metrics if not s.get("rejected", True)]) / len(creativity_metrics)
                    }
                else:
                    states["creativity"] = self._get_default_module_state("creativity")
            except:
                states["creativity"] = self._get_default_module_state("creativity")
            
            # Reflection (Step 30)
            try:
                reflection_cycle = await self.db.llm_reflection_log.find_one(
                    sort=[("timestamp", -1)]
                )
                reflection_params = await self.db.llm_learning_parameters.find_one(
                    sort=[("timestamp", -1)]
                )
                if reflection_cycle and reflection_params:
                    states["reflection"] = {
                        "novelty_weight": reflection_params.get("novelty_weight", 0.60),
                        "stability_weight": reflection_params.get("stability_weight", 0.50),
                        "ethical_threshold": reflection_params.get("ethical_threshold", 0.75),
                        "learning_health": reflection_cycle.get("learning_health_index", 0.75),
                        "performance": reflection_cycle.get("overall_performance_score", 70.0) / 100.0,
                        "ethical_status": reflection_cycle.get("ethical_alignment_status", "good")
                    }
                else:
                    states["reflection"] = self._get_default_module_state("reflection")
            except:
                states["reflection"] = self._get_default_module_state("reflection")
            
            # Memory (Step 31)
            try:
                active_nodes = await self.db.llm_memory_nodes.count_documents({
                    "decay_weight": {"$gte": 0.20}
                })
                latest_fusion = await self.db.llm_memory_trace.find_one({
                    "trace_type": "fusion_cycle"
                }, sort=[("timestamp", -1)])
                
                if latest_fusion:
                    fusion_metrics = latest_fusion.get("fusion_metrics", {})
                    states["memory"] = {
                        "active_nodes": active_nodes,
                        "retention_index": fusion_metrics.get("memory_retention_index", 0.85),
                        "fusion_efficiency": fusion_metrics.get("fusion_efficiency", 0.80),
                        "ethical_continuity": fusion_metrics.get("ethical_continuity", 0.92),
                        "persistence_health": fusion_metrics.get("persistence_health", 0.88)
                    }
                else:
                    states["memory"] = self._get_default_module_state("memory")
            except:
                states["memory"] = self._get_default_module_state("memory")
            
            # Cohesion (Step 32)
            try:
                cohesion_report = await self.db.llm_cohesion_reports.find_one(
                    sort=[("timestamp", -1)]
                )
                if cohesion_report:
                    metrics = cohesion_report.get("metrics", {})
                    states["cohesion"] = {
                        "alignment_score": metrics.get("alignment_score", 0.85),
                        "system_health": metrics.get("system_health_index", 0.80),
                        "ethical_continuity": metrics.get("ethical_continuity", 0.90),
                        "parameter_harmony": metrics.get("parameter_harmony_score", 0.85),
                        "drift_detected": metrics.get("drift_detected", False)
                    }
                else:
                    states["cohesion"] = self._get_default_module_state("cohesion")
            except:
                states["cohesion"] = self._get_default_module_state("cohesion")
            
            # Ethics (Step 33)
            try:
                ethics_metrics = await self.db.llm_ethics_metrics.find_one(
                    sort=[("timestamp", -1)]
                )
                if ethics_metrics:
                    states["ethics"] = {
                        "compliance_index": ethics_metrics.get("compliance_index", 0.95),
                        "ethical_continuity": ethics_metrics.get("ethical_continuity", 0.93),
                        "fair_play_score": ethics_metrics.get("fair_play_score", 0.90),
                        "transparency_score": ethics_metrics.get("transparency_score", 0.88),
                        "status": ethics_metrics.get("status", "good")
                    }
                else:
                    states["ethics"] = self._get_default_module_state("ethics")
            except:
                states["ethics"] = self._get_default_module_state("ethics")
            
            return states
            
        except Exception as e:
            logger.error(f"Error gathering module states: {e}")
            return {}
    
    def _get_default_module_state(self, module_name: str) -> Dict[str, Any]:
        """Get default state for a module"""
        defaults = {
            "creativity": {
                "avg_novelty": 0.70,
                "avg_stability": 0.65,
                "avg_ethical": 0.88,
                "count": 0,
                "health": 0.75
            },
            "reflection": {
                "novelty_weight": 0.60,
                "stability_weight": 0.50,
                "ethical_threshold": 0.75,
                "learning_health": 0.75,
                "performance": 0.70,
                "ethical_status": "good"
            },
            "memory": {
                "active_nodes": 0,
                "retention_index": 0.85,
                "fusion_efficiency": 0.80,
                "ethical_continuity": 0.92,
                "persistence_health": 0.88
            },
            "cohesion": {
                "alignment_score": 0.85,
                "system_health": 0.80,
                "ethical_continuity": 0.90,
                "parameter_harmony": 0.85,
                "drift_detected": False
            },
            "ethics": {
                "compliance_index": 0.95,
                "ethical_continuity": 0.93,
                "fair_play_score": 0.90,
                "transparency_score": 0.88,
                "status": "good"
            }
        }
        return defaults.get(module_name, {})
    
    async def _calculate_cross_module_alignments(
        self, 
        module_states: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate alignment scores between adjacent modules"""
        alignments = {}
        
        # Creativity <-> Reflection
        if "creativity" in module_states and "reflection" in module_states:
            creativity = module_states["creativity"]
            reflection = module_states["reflection"]
            
            novelty_delta = abs(creativity.get("avg_novelty", 0.7) - reflection.get("novelty_weight", 0.6))
            stability_delta = abs(creativity.get("avg_stability", 0.65) - reflection.get("stability_weight", 0.5))
            
            alignments["creativity_reflection"] = 1.0 - ((novelty_delta + stability_delta) / 2.0)
        else:
            alignments["creativity_reflection"] = 0.85
        
        # Reflection <-> Memory
        if "reflection" in module_states and "memory" in module_states:
            reflection = module_states["reflection"]
            memory = module_states["memory"]
            
            learning_delta = abs(reflection.get("learning_health", 0.75) - memory.get("persistence_health", 0.88))
            
            alignments["reflection_memory"] = 1.0 - learning_delta
        else:
            alignments["reflection_memory"] = 0.88
        
        # Memory <-> Cohesion
        if "memory" in module_states and "cohesion" in module_states:
            memory = module_states["memory"]
            cohesion = module_states["cohesion"]
            
            health_delta = abs(memory.get("persistence_health", 0.88) - cohesion.get("system_health", 0.80))
            ethical_delta = abs(memory.get("ethical_continuity", 0.92) - cohesion.get("ethical_continuity", 0.90))
            
            alignments["memory_cohesion"] = 1.0 - ((health_delta + ethical_delta) / 2.0)
        else:
            alignments["memory_cohesion"] = 0.90
        
        # Cohesion <-> Ethics
        if "cohesion" in module_states and "ethics" in module_states:
            cohesion = module_states["cohesion"]
            ethics = module_states["ethics"]
            
            ethical_delta = abs(cohesion.get("ethical_continuity", 0.90) - ethics.get("ethical_continuity", 0.93))
            alignment_delta = abs(cohesion.get("alignment_score", 0.85) - ethics.get("compliance_index", 0.95))
            
            alignments["cohesion_ethics"] = 1.0 - ((ethical_delta + alignment_delta) / 2.0)
        else:
            alignments["cohesion_ethics"] = 0.92
        
        # Ethics <-> Creativity (completing the cycle)
        if "ethics" in module_states and "creativity" in module_states:
            ethics = module_states["ethics"]
            creativity = module_states["creativity"]
            
            ethical_delta = abs(ethics.get("compliance_index", 0.95) - creativity.get("avg_ethical", 0.88))
            
            alignments["ethics_creativity"] = 1.0 - ethical_delta
        else:
            alignments["ethics_creativity"] = 0.90
        
        return alignments
    
    async def _calculate_drift_velocity(self) -> float:
        """
        Temporal Stability Monitor (TSM) - Calculate rate of parameter change
        """
        try:
            # Get recent parameter history from reflection module
            recent_params = await self.db.llm_learning_parameters.find().sort(
                "timestamp", -1
            ).limit(5).to_list(5)
            
            if len(recent_params) < 2:
                return 0.0
            
            # Calculate velocity (change rate) for key parameters
            velocities = []
            for i in range(len(recent_params) - 1):
                current = recent_params[i]
                previous = recent_params[i + 1]
                
                novelty_velocity = abs(current.get("novelty_weight", 0.6) - previous.get("novelty_weight", 0.6))
                stability_velocity = abs(current.get("stability_weight", 0.5) - previous.get("stability_weight", 0.5))
                
                velocities.append((novelty_velocity + stability_velocity) / 2.0)
            
            return np.mean(velocities) if velocities else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating drift velocity: {e}")
            return 0.0
    
    async def _calculate_oscillation_amplitude(self) -> float:
        """
        Temporal Stability Monitor (TSM) - Calculate degree of parameter fluctuation
        """
        try:
            # Get recent parameter history
            recent_params = await self.db.llm_learning_parameters.find().sort(
                "timestamp", -1
            ).limit(10).to_list(10)
            
            if len(recent_params) < 3:
                return 0.0
            
            # Calculate standard deviation (fluctuation) for key parameters
            novelty_values = [p.get("novelty_weight", 0.6) for p in recent_params]
            stability_values = [p.get("stability_weight", 0.5) for p in recent_params]
            
            novelty_std = np.std(novelty_values)
            stability_std = np.std(stability_values)
            
            return (novelty_std + stability_std) / 2.0
            
        except Exception as e:
            logger.error(f"Error calculating oscillation amplitude: {e}")
            return 0.0
    
    async def _calculate_convergence_score(
        self, 
        module_states: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate how close system is to equilibrium"""
        # Check if all modules are within acceptable ranges
        scores = []
        
        # Creativity convergence
        if "creativity" in module_states:
            novelty = module_states["creativity"].get("avg_novelty", 0.7)
            # Target range: 0.60-0.75
            if 0.60 <= novelty <= 0.75:
                scores.append(1.0)
            else:
                scores.append(max(0.0, 1.0 - abs(novelty - 0.675) / 0.15))
        
        # Reflection convergence
        if "reflection" in module_states:
            learning_health = module_states["reflection"].get("learning_health", 0.75)
            # Target range: 0.70-0.85
            if 0.70 <= learning_health <= 0.85:
                scores.append(1.0)
            else:
                scores.append(max(0.0, 1.0 - abs(learning_health - 0.775) / 0.15))
        
        # Memory convergence
        if "memory" in module_states:
            persistence_health = module_states["memory"].get("persistence_health", 0.88)
            # Target: ≥ 0.88
            if persistence_health >= 0.88:
                scores.append(1.0)
            else:
                scores.append(max(0.0, persistence_health / 0.88))
        
        # Cohesion convergence
        if "cohesion" in module_states:
            alignment = module_states["cohesion"].get("alignment_score", 0.85)
            # Target: ≥ 0.90
            if alignment >= 0.90:
                scores.append(1.0)
            else:
                scores.append(max(0.0, alignment / 0.90))
        
        # Ethics convergence
        if "ethics" in module_states:
            compliance = module_states["ethics"].get("compliance_index", 0.95)
            # Target: ≥ 0.95
            if compliance >= 0.95:
                scores.append(1.0)
            else:
                scores.append(max(0.0, compliance / 0.95))
        
        return np.mean(scores) if scores else 0.85
    
    async def _calculate_module_resonance(
        self, 
        module_states: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate resonance score for each module"""
        resonance = {}
        
        # Creativity resonance
        if "creativity" in module_states:
            creativity = module_states["creativity"]
            resonance["creativity"] = (
                creativity.get("health", 0.75) * 0.4 +
                creativity.get("avg_ethical", 0.88) * 0.3 +
                creativity.get("avg_stability", 0.65) * 0.3
            )
        else:
            resonance["creativity"] = 0.75
        
        # Reflection resonance
        if "reflection" in module_states:
            reflection = module_states["reflection"]
            resonance["reflection"] = (
                reflection.get("learning_health", 0.75) * 0.4 +
                reflection.get("performance", 0.70) * 0.3 +
                (1.0 if reflection.get("ethical_status") in ["excellent", "good"] else 0.7) * 0.3
            )
        else:
            resonance["reflection"] = 0.75
        
        # Memory resonance
        if "memory" in module_states:
            memory = module_states["memory"]
            resonance["memory"] = (
                memory.get("persistence_health", 0.88) * 0.4 +
                memory.get("retention_index", 0.85) * 0.3 +
                memory.get("ethical_continuity", 0.92) * 0.3
            )
        else:
            resonance["memory"] = 0.88
        
        # Cohesion resonance
        if "cohesion" in module_states:
            cohesion = module_states["cohesion"]
            resonance["cohesion"] = (
                cohesion.get("alignment_score", 0.85) * 0.4 +
                cohesion.get("system_health", 0.80) * 0.3 +
                cohesion.get("parameter_harmony", 0.85) * 0.3
            )
        else:
            resonance["cohesion"] = 0.85
        
        # Ethics resonance
        if "ethics" in module_states:
            ethics = module_states["ethics"]
            resonance["ethics"] = (
                ethics.get("compliance_index", 0.95) * 0.5 +
                ethics.get("ethical_continuity", 0.93) * 0.5
            )
        else:
            resonance["ethics"] = 0.94
        
        return resonance
    
    async def _calculate_resonance_index(
        self,
        alignments: Dict[str, float],
        module_resonance: Dict[str, float],
        convergence_score: float
    ) -> float:
        """Calculate overall resonance index"""
        # Component 1: Cross-module alignments (40%)
        avg_alignment = np.mean(list(alignments.values()))
        
        # Component 2: Module resonance (40%)
        avg_module_resonance = np.mean(list(module_resonance.values()))
        
        # Component 3: Convergence (20%)
        
        # Weighted combination
        resonance_index = (
            avg_alignment * 0.40 +
            avg_module_resonance * 0.40 +
            convergence_score * 0.20
        )
        
        return min(1.0, max(0.0, resonance_index))
    
    async def _calculate_temporal_stability(
        self,
        drift_velocity: float,
        oscillation_amplitude: float
    ) -> float:
        """
        Temporal Stability Monitor (TSM) - Calculate overall temporal stability
        """
        # Stability is inverse of drift and oscillation
        # If drift_velocity is low and oscillation is low, stability is high
        
        drift_stability = max(0.0, 1.0 - (drift_velocity / self.stability_thresholds["max_drift_velocity"]))
        oscillation_stability = max(0.0, 1.0 - (oscillation_amplitude / self.stability_thresholds["max_oscillation_amplitude"]))
        
        # Weighted combination
        temporal_stability = (drift_stability * 0.5 + oscillation_stability * 0.5)
        
        return min(1.0, max(0.0, temporal_stability))
    
    async def _calculate_feedback_equilibrium(
        self, 
        module_states: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Adaptive Feedback Regulator (AFR) - Calculate feedback weight balance
        """
        # Target equilibrium: novelty and stability weights should be balanced
        # Target: 0.50 ± 0.05 (meaning novelty:stability ratio close to 1:1)
        
        if "reflection" in module_states:
            novelty_weight = module_states["reflection"].get("novelty_weight", 0.60)
            stability_weight = module_states["reflection"].get("stability_weight", 0.50)
            
            total_weight = novelty_weight + stability_weight
            if total_weight > 0:
                novelty_ratio = novelty_weight / total_weight
                # Target is 0.50 (balanced)
                deviation = abs(novelty_ratio - 0.50)
                equilibrium = max(0.0, 1.0 - (deviation / 0.25))  # Normalize to 0-1
                return min(1.0, equilibrium)
        
        return 0.50
    
    async def _calculate_entropy_balance(
        self, 
        module_states: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Entropy Control Mechanism (ECM) - Calculate novelty vs stability balance
        Target range: 0.45-0.55 (balanced entropy)
        """
        if "creativity" in module_states:
            novelty = module_states["creativity"].get("avg_novelty", 0.70)
            stability = module_states["creativity"].get("avg_stability", 0.65)
            
            # Entropy balance is the ratio favoring novelty
            # Higher novelty = higher entropy (more exploration)
            # Higher stability = lower entropy (more exploitation)
            
            total = novelty + stability
            if total > 0:
                entropy_score = novelty / total  # Will be 0-1
                
                # Check if within target range (0.45-0.55)
                if self.target_metrics["entropy_balance_min"] <= entropy_score <= self.target_metrics["entropy_balance_max"]:
                    return entropy_score
                else:
                    # Return how far from optimal
                    if entropy_score < 0.45:
                        return entropy_score  # Below range
                    else:
                        return min(0.65, entropy_score)  # Above range but capped
        
        return 0.50  # Default balanced
    
    def _determine_resonance_health(
        self,
        resonance_index: float,
        temporal_stability: float,
        feedback_equilibrium: float,
        entropy_balance: float
    ) -> Tuple[str, List[str]]:
        """Determine overall resonance health and generate warnings"""
        warnings = []
        
        # Check resonance index
        if resonance_index < self.target_metrics["resonance_index"]:
            warnings.append(
                f"Resonance index ({resonance_index:.3f}) below target "
                f"({self.target_metrics['resonance_index']:.2f})"
            )
        
        # Check temporal stability
        if temporal_stability < self.target_metrics["temporal_stability"]:
            warnings.append(
                f"Temporal stability ({temporal_stability:.3f}) below target "
                f"({self.target_metrics['temporal_stability']:.2f})"
            )
        
        # Check feedback equilibrium
        target_equilibrium = self.target_metrics["feedback_equilibrium"]
        tolerance = self.target_metrics["feedback_equilibrium_tolerance"]
        if not (target_equilibrium - tolerance <= feedback_equilibrium <= target_equilibrium + tolerance):
            warnings.append(
                f"Feedback equilibrium ({feedback_equilibrium:.3f}) outside target range "
                f"({target_equilibrium - tolerance:.2f}-{target_equilibrium + tolerance:.2f})"
            )
        
        # Check entropy balance
        if not (self.target_metrics["entropy_balance_min"] <= entropy_balance <= self.target_metrics["entropy_balance_max"]):
            warnings.append(
                f"Entropy balance ({entropy_balance:.3f}) outside target range "
                f"({self.target_metrics['entropy_balance_min']:.2f}-{self.target_metrics['entropy_balance_max']:.2f})"
            )
        
        # Determine health status
        metrics_in_target = sum([
            resonance_index >= self.target_metrics["resonance_index"],
            temporal_stability >= self.target_metrics["temporal_stability"],
            (target_equilibrium - tolerance <= feedback_equilibrium <= target_equilibrium + tolerance),
            (self.target_metrics["entropy_balance_min"] <= entropy_balance <= self.target_metrics["entropy_balance_max"])
        ])
        
        if metrics_in_target == 4:
            health = "excellent"
        elif metrics_in_target >= 3:
            health = "good"
        elif metrics_in_target >= 2:
            health = "moderate"
        else:
            health = "needs_attention"
        
        return health, warnings
    
    async def stability_forecast(
        self,
        horizon_hours: int = 24
    ) -> StabilityForecast:
        """
        Temporal Stability Monitor (TSM) - Predict potential drift over upcoming sessions
        
        Args:
            horizon_hours: How many hours ahead to forecast
        
        Returns:
            Stability forecast with predictions and risk assessment
        """
        try:
            logger.info(f"Generating stability forecast for next {horizon_hours} hours...")
            
            # Get recent resonance metrics
            recent_metrics = await self.db.llm_resonance_metrics.find().sort(
                "timestamp", -1
            ).limit(10).to_list(10)
            
            if len(recent_metrics) < 3:
                # Insufficient data for forecasting
                return self._generate_default_forecast(horizon_hours)
            
            # Extract trends
            resonance_values = [m.get("resonance_index", 0.90) for m in recent_metrics]
            stability_values = [m.get("temporal_stability", 0.88) for m in recent_metrics]
            entropy_values = [m.get("entropy_balance", 0.50) for m in recent_metrics]
            
            # Simple linear extrapolation for prediction
            predicted_resonance = self._extrapolate_trend(resonance_values)
            predicted_stability = self._extrapolate_trend(stability_values)
            predicted_entropy = self._extrapolate_trend(entropy_values)
            
            # Determine trends
            resonance_trend = self._classify_trend(resonance_values)
            stability_trend = self._classify_trend(stability_values)
            entropy_trend = self._classify_trend(entropy_values)
            
            # Risk assessment
            drift_risk = await self._assess_drift_risk()
            oscillation_risk = await self._assess_oscillation_risk()
            stagnation_risk = await self._assess_stagnation_risk(entropy_values)
            
            # Calculate confidence
            confidence = self._calculate_forecast_confidence(len(recent_metrics))
            
            # Generate recommendations
            recommendations = self._generate_forecast_recommendations(
                predicted_resonance, predicted_stability, predicted_entropy,
                drift_risk, oscillation_risk, stagnation_risk
            )
            
            forecast = StabilityForecast(
                forecast_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                forecast_horizon_hours=horizon_hours,
                predicted_resonance_index=round(predicted_resonance, 3),
                predicted_temporal_stability=round(predicted_stability, 3),
                predicted_entropy_balance=round(predicted_entropy, 3),
                resonance_trend=resonance_trend,
                stability_trend=stability_trend,
                entropy_trend=entropy_trend,
                drift_risk_level=drift_risk,
                oscillation_risk_level=oscillation_risk,
                stagnation_risk_level=stagnation_risk,
                confidence_score=confidence,
                recommended_interventions=recommendations
            )
            
            # Store forecast
            await self.db.llm_resonance_forecast.insert_one(forecast.to_dict())
            
            logger.info(
                f"Forecast generated: Resonance={predicted_resonance:.3f}, "
                f"Stability={predicted_stability:.3f}, Confidence={confidence:.2f}"
            )
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error generating stability forecast: {e}")
            return self._generate_default_forecast(horizon_hours)
    
    def _extrapolate_trend(self, values: List[float]) -> float:
        """Simple linear extrapolation"""
        if len(values) < 2:
            return values[0] if values else 0.85
        
        # Calculate simple trend
        recent = values[:3]  # Most recent 3
        older = values[3:6] if len(values) > 3 else values[-3:]
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        # Extrapolate
        delta = recent_avg - older_avg
        predicted = recent_avg + delta
        
        # Clamp to valid range
        return max(0.0, min(1.0, predicted))
    
    def _classify_trend(self, values: List[float]) -> str:
        """Classify trend direction"""
        if len(values) < 2:
            return "stable"
        
        recent = np.mean(values[:3])
        older = np.mean(values[3:]) if len(values) > 3 else values[-1]
        
        delta = recent - older
        
        if delta > 0.02:
            return "improving"
        elif delta < -0.02:
            return "declining"
        else:
            return "stable"
    
    async def _assess_drift_risk(self) -> str:
        """Assess risk of parameter drift"""
        drift_velocity = await self._calculate_drift_velocity()
        
        if drift_velocity > self.stability_thresholds["max_drift_velocity"] * 1.5:
            return "critical"
        elif drift_velocity > self.stability_thresholds["max_drift_velocity"]:
            return "high"
        elif drift_velocity > self.stability_thresholds["max_drift_velocity"] * 0.5:
            return "medium"
        else:
            return "low"
    
    async def _assess_oscillation_risk(self) -> str:
        """Assess risk of parameter oscillation"""
        oscillation = await self._calculate_oscillation_amplitude()
        
        if oscillation > self.stability_thresholds["max_oscillation_amplitude"] * 1.5:
            return "critical"
        elif oscillation > self.stability_thresholds["max_oscillation_amplitude"]:
            return "high"
        elif oscillation > self.stability_thresholds["max_oscillation_amplitude"] * 0.5:
            return "medium"
        else:
            return "low"
    
    async def _assess_stagnation_risk(self, entropy_values: List[float]) -> str:
        """Assess risk of system stagnation (too little novelty)"""
        if not entropy_values:
            return "low"
        
        avg_entropy = np.mean(entropy_values)
        
        if avg_entropy < 0.35:
            return "critical"
        elif avg_entropy < 0.40:
            return "high"
        elif avg_entropy < 0.45:
            return "medium"
        else:
            return "low"
    
    def _calculate_forecast_confidence(self, data_points: int) -> float:
        """Calculate confidence in forecast based on available data"""
        # More data points = higher confidence
        base_confidence = min(0.95, 0.50 + (data_points * 0.05))
        return round(base_confidence, 2)
    
    def _generate_forecast_recommendations(
        self,
        predicted_resonance: float,
        predicted_stability: float,
        predicted_entropy: float,
        drift_risk: str,
        oscillation_risk: str,
        stagnation_risk: str
    ) -> List[str]:
        """Generate recommended interventions based on forecast"""
        recommendations = []
        
        # Resonance predictions
        if predicted_resonance < self.target_metrics["resonance_index"]:
            recommendations.append(
                f"Predicted resonance ({predicted_resonance:.3f}) below target - "
                "consider increasing cross-module synchronization"
            )
        
        # Stability predictions
        if predicted_stability < self.target_metrics["temporal_stability"]:
            recommendations.append(
                f"Predicted stability ({predicted_stability:.3f}) below target - "
                "reduce parameter change rates"
            )
        
        # Entropy predictions
        if predicted_entropy < 0.45:
            recommendations.append(
                "Low entropy predicted - increase novelty exploration to prevent stagnation"
            )
        elif predicted_entropy > 0.55:
            recommendations.append(
                "High entropy predicted - increase stability focus to prevent instability"
            )
        
        # Risk-based recommendations
        if drift_risk in ["critical", "high"]:
            recommendations.append(f"⚠️ {drift_risk.upper()} drift risk - implement drift correction mechanisms")
        
        if oscillation_risk in ["critical", "high"]:
            recommendations.append(f"⚠️ {oscillation_risk.upper()} oscillation risk - stabilize parameter updates")
        
        if stagnation_risk in ["critical", "high"]:
            recommendations.append(f"⚠️ {stagnation_risk.upper()} stagnation risk - increase creative exploration")
        
        if not recommendations:
            recommendations.append("✅ System forecast appears stable - continue current approach")
        
        return recommendations[:5]
    
    def _generate_default_forecast(self, horizon_hours: int) -> StabilityForecast:
        """Generate default forecast when insufficient data"""
        return StabilityForecast(
            forecast_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            forecast_horizon_hours=horizon_hours,
            predicted_resonance_index=0.90,
            predicted_temporal_stability=0.88,
            predicted_entropy_balance=0.50,
            resonance_trend="stable",
            stability_trend="stable",
            entropy_trend="stable",
            drift_risk_level="low",
            oscillation_risk_level="low",
            stagnation_risk_level="low",
            confidence_score=0.50,
            recommended_interventions=["Insufficient data for detailed forecast - continue monitoring"]
        )
    
    async def balance_feedback_weights(
        self,
        force_recalibration: bool = False
    ) -> Dict[str, Any]:
        """
        Adaptive Feedback Regulator (AFR) - Adjust learning feedback gains in real time
        
        Balances novelty, stability, and ethical weights to maintain equilibrium.
        Operates in advisory mode - does not apply changes without approval.
        
        Args:
            force_recalibration: Force recalibration even if system is balanced
        
        Returns:
            Balance adjustment results
        """
        try:
            logger.info("AFR: Balancing feedback weights...")
            
            # Get current resonance state
            current_metrics = await self.analyze_resonance_state()
            
            # Check if recalibration needed
            needs_recalibration = (
                force_recalibration or
                current_metrics.feedback_equilibrium < 0.45 or
                current_metrics.feedback_equilibrium > 0.55 or
                current_metrics.entropy_balance < 0.45 or
                current_metrics.entropy_balance > 0.55
            )
            
            if not needs_recalibration:
                return {
                    "recalibration_needed": False,
                    "message": "System feedback weights are balanced",
                    "current_equilibrium": current_metrics.feedback_equilibrium,
                    "current_entropy": current_metrics.entropy_balance
                }
            
            # Get current parameters
            module_states = await self._gather_module_states()
            reflection = module_states.get("reflection", {})
            
            current_novelty = reflection.get("novelty_weight", 0.60)
            current_stability = reflection.get("stability_weight", 0.50)
            current_ethical = reflection.get("ethical_threshold", 0.75)
            
            # Calculate recommended adjustments
            adjustments = {}
            
            # Adjust for feedback equilibrium
            if current_metrics.feedback_equilibrium < 0.45:
                # Too much emphasis on stability - increase novelty
                adjustments["novelty_weight"] = min(0.80, current_novelty + 0.03)
                adjustments["stability_weight"] = max(0.40, current_stability - 0.02)
            elif current_metrics.feedback_equilibrium > 0.55:
                # Too much emphasis on novelty - increase stability
                adjustments["novelty_weight"] = max(0.50, current_novelty - 0.02)
                adjustments["stability_weight"] = min(0.70, current_stability + 0.03)
            
            # Adjust for entropy balance
            if current_metrics.entropy_balance < 0.45:
                # Stagnation risk - boost novelty
                if "novelty_weight" not in adjustments:
                    adjustments["novelty_weight"] = min(0.80, current_novelty + 0.04)
            elif current_metrics.entropy_balance > 0.55:
                # Instability risk - boost stability
                if "stability_weight" not in adjustments:
                    adjustments["stability_weight"] = min(0.70, current_stability + 0.04)
            
            # Ethical threshold adjustment
            if current_metrics.ethics_resonance < 0.90:
                adjustments["ethical_threshold"] = min(0.95, current_ethical + 0.02)
            
            result = {
                "recalibration_needed": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "current_parameters": {
                    "novelty_weight": current_novelty,
                    "stability_weight": current_stability,
                    "ethical_threshold": current_ethical
                },
                "recommended_adjustments": adjustments,
                "current_equilibrium": current_metrics.feedback_equilibrium,
                "current_entropy": current_metrics.entropy_balance,
                "target_equilibrium": self.target_metrics["feedback_equilibrium"],
                "target_entropy_range": [
                    self.target_metrics["entropy_balance_min"],
                    self.target_metrics["entropy_balance_max"]
                ],
                "advisory_note": "Adjustments are advisory only and require human approval via Ethics Layer (Step 33)"
            }
            
            # Store balance action (make a copy to avoid _id in response)
            await self.db.llm_resonance_balance_actions.insert_one(result.copy())
            
            logger.info(
                f"AFR: Balance recommendations generated with {len(adjustments)} adjustments"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error balancing feedback weights: {e}")
            return {"error": str(e)}
    
    async def record_resonance_snapshot(self) -> ResonanceSnapshot:
        """
        Persistence Resonator (PR) - Store longitudinal metrics in DB
        
        Returns:
            Stored resonance snapshot
        """
        try:
            logger.info("PR: Recording resonance snapshot...")
            
            # Get current module states
            module_states = await self._gather_module_states()
            
            # Get current metrics
            current_metrics = await self.db.llm_resonance_metrics.find_one(
                sort=[("timestamp", -1)]
            )
            
            if current_metrics:
                resonance_metrics = {
                    "resonance_index": current_metrics.get("resonance_index"),
                    "temporal_stability": current_metrics.get("temporal_stability"),
                    "feedback_equilibrium": current_metrics.get("feedback_equilibrium"),
                    "entropy_balance": current_metrics.get("entropy_balance")
                }
            else:
                resonance_metrics = {
                    "resonance_index": 0.90,
                    "temporal_stability": 0.88,
                    "feedback_equilibrium": 0.50,
                    "entropy_balance": 0.50
                }
            
            # Count recent activity
            game_sessions = 0  # Would count from game log
            ethics_scans = await self.db.llm_ethics_metrics.count_documents({
                "timestamp": {"$gte": (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()}
            })
            cohesion_cycles = await self.db.llm_cohesion_reports.count_documents({
                "timestamp": {"$gte": (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()}
            })
            
            # Identify notable events
            events = []
            if current_metrics and current_metrics.get("resonance_health") == "needs_attention":
                events.append("Resonance health requires attention")
            if current_metrics and len(current_metrics.get("stability_warnings", [])) > 0:
                events.extend(current_metrics.get("stability_warnings", []))
            
            snapshot = ResonanceSnapshot(
                snapshot_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                module_states=module_states,
                resonance_metrics=resonance_metrics,
                game_sessions_count=game_sessions,
                ethics_scans_count=ethics_scans,
                cohesion_cycles_count=cohesion_cycles,
                events=events
            )
            
            # Store snapshot
            await self.db.llm_resonance_snapshots.insert_one(snapshot.to_dict())
            
            logger.info(f"PR: Resonance snapshot recorded: {snapshot.snapshot_id}")
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error recording resonance snapshot: {e}")
            raise
    
    async def generate_resonance_report(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> ResonanceReport:
        """
        Generate comprehensive Resonance Report #001 (or subsequent reports)
        
        Args:
            start_date: Report period start (ISO format)
            end_date: Report period end (ISO format)
        
        Returns:
            Complete resonance report
        """
        try:
            # Increment report counter
            self.report_counter += 1
            report_number = self.report_counter
            
            logger.info(f"Generating Resonance Report #{report_number:03d}...")
            
            # Default to last 7 days if not specified
            if not end_date:
                end_date = datetime.now(timezone.utc).isoformat()
            if not start_date:
                start_date = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            
            # Get metrics in period
            metrics_in_period = await self.db.llm_resonance_metrics.find({
                "timestamp": {"$gte": start_date, "$lte": end_date}
            }).to_list(1000)
            
            # Calculate summary metrics
            if metrics_in_period:
                overall_resonance = np.mean([m.get("resonance_index", 0.90) for m in metrics_in_period])
                overall_stability = np.mean([m.get("temporal_stability", 0.88) for m in metrics_in_period])
                overall_equilibrium = np.mean([m.get("feedback_equilibrium", 0.50) for m in metrics_in_period])
                overall_entropy = np.mean([m.get("entropy_balance", 0.50) for m in metrics_in_period])
            else:
                overall_resonance = 0.90
                overall_stability = 0.88
                overall_equilibrium = 0.50
                overall_entropy = 0.50
            
            # Module resonance scores
            module_scores = {}
            if metrics_in_period:
                module_scores = {
                    "creativity": np.mean([m.get("creativity_resonance", 0.75) for m in metrics_in_period]),
                    "reflection": np.mean([m.get("reflection_resonance", 0.75) for m in metrics_in_period]),
                    "memory": np.mean([m.get("memory_resonance", 0.88) for m in metrics_in_period]),
                    "cohesion": np.mean([m.get("cohesion_resonance", 0.85) for m in metrics_in_period]),
                    "ethics": np.mean([m.get("ethics_resonance", 0.94) for m in metrics_in_period])
                }
            
            # Module alignment matrix
            alignment_matrix = {}
            if metrics_in_period:
                latest = metrics_in_period[0]
                alignment_matrix = {
                    "creativity_reflection": latest.get("creativity_reflection_alignment", 0.85),
                    "reflection_memory": latest.get("reflection_memory_alignment", 0.88),
                    "memory_cohesion": latest.get("memory_cohesion_alignment", 0.90),
                    "cohesion_ethics": latest.get("cohesion_ethics_alignment", 0.92),
                    "ethics_creativity": latest.get("ethics_creativity_alignment", 0.90)
                }
            
            # Drift analysis
            drift_analysis = {
                "avg_drift_velocity": np.mean([m.get("drift_velocity", 0.0) for m in metrics_in_period]) if metrics_in_period else 0.0,
                "max_drift_velocity": max([m.get("drift_velocity", 0.0) for m in metrics_in_period]) if metrics_in_period else 0.0,
                "drift_threshold": self.stability_thresholds["max_drift_velocity"],
                "drift_violations": sum(1 for m in metrics_in_period if m.get("drift_velocity", 0.0) > self.stability_thresholds["max_drift_velocity"]) if metrics_in_period else 0
            }
            
            # Oscillation analysis
            oscillation_analysis = {
                "avg_oscillation": np.mean([m.get("oscillation_amplitude", 0.0) for m in metrics_in_period]) if metrics_in_period else 0.0,
                "max_oscillation": max([m.get("oscillation_amplitude", 0.0) for m in metrics_in_period]) if metrics_in_period else 0.0,
                "oscillation_threshold": self.stability_thresholds["max_oscillation_amplitude"],
                "oscillation_violations": sum(1 for m in metrics_in_period if m.get("oscillation_amplitude", 0.0) > self.stability_thresholds["max_oscillation_amplitude"]) if metrics_in_period else 0
            }
            
            # Entropy analysis
            entropy_analysis = {
                "avg_entropy": overall_entropy,
                "min_entropy": min([m.get("entropy_balance", 0.50) for m in metrics_in_period]) if metrics_in_period else 0.50,
                "max_entropy": max([m.get("entropy_balance", 0.50) for m in metrics_in_period]) if metrics_in_period else 0.50,
                "target_range": [self.target_metrics["entropy_balance_min"], self.target_metrics["entropy_balance_max"]],
                "in_range_count": sum(1 for m in metrics_in_period if self.target_metrics["entropy_balance_min"] <= m.get("entropy_balance", 0.50) <= self.target_metrics["entropy_balance_max"]) if metrics_in_period else 0
            }
            
            # Generate forecasts
            short_term_forecast = await self.stability_forecast(horizon_hours=24)
            long_term_forecast = await self.stability_forecast(horizon_hours=168)  # 7 days
            
            # Trajectories
            resonance_trajectory = [m.get("resonance_index", 0.90) for m in metrics_in_period[-20:]] if metrics_in_period else []
            stability_trajectory = [m.get("temporal_stability", 0.88) for m in metrics_in_period[-20:]] if metrics_in_period else []
            entropy_trajectory = [m.get("entropy_balance", 0.50) for m in metrics_in_period[-20:]] if metrics_in_period else []
            
            # Generate recommendations
            recommendations = self._generate_report_recommendations(
                overall_resonance, overall_stability, overall_equilibrium, overall_entropy,
                drift_analysis, oscillation_analysis, entropy_analysis
            )
            
            # Intervention priorities
            intervention_priorities = self._prioritize_interventions(
                overall_resonance, overall_stability, overall_equilibrium, overall_entropy,
                drift_analysis, oscillation_analysis
            )
            
            # Health assessment
            health_summary = await self._generate_health_summary(
                overall_resonance, overall_stability, overall_equilibrium, overall_entropy
            )
            
            # Target compliance (convert to bool to avoid numpy bool serialization issues)
            target_compliance = {
                "resonance_index": bool(overall_resonance >= self.target_metrics["resonance_index"]),
                "temporal_stability": bool(overall_stability >= self.target_metrics["temporal_stability"]),
                "feedback_equilibrium": bool(abs(overall_equilibrium - self.target_metrics["feedback_equilibrium"]) <= self.target_metrics["feedback_equilibrium_tolerance"]),
                "entropy_balance": bool(self.target_metrics["entropy_balance_min"] <= overall_entropy <= self.target_metrics["entropy_balance_max"])
            }
            
            report = ResonanceReport(
                report_id=str(uuid.uuid4()),
                report_number=report_number,
                timestamp=datetime.now(timezone.utc).isoformat(),
                reporting_period={"start": start_date, "end": end_date},
                overall_resonance_index=round(overall_resonance, 3),
                overall_temporal_stability=round(overall_stability, 3),
                overall_feedback_equilibrium=round(overall_equilibrium, 3),
                overall_entropy_balance=round(overall_entropy, 3),
                module_resonance_scores={k: round(v, 3) for k, v in module_scores.items()},
                module_alignment_matrix=alignment_matrix,
                drift_analysis=drift_analysis,
                oscillation_analysis=oscillation_analysis,
                entropy_analysis=entropy_analysis,
                short_term_forecast=short_term_forecast.to_dict(),
                long_term_forecast=long_term_forecast.to_dict(),
                resonance_trajectory=resonance_trajectory,
                stability_trajectory=stability_trajectory,
                entropy_trajectory=entropy_trajectory,
                recommendations=recommendations,
                intervention_priorities=intervention_priorities,
                system_health_summary=health_summary,
                target_compliance=target_compliance
            )
            
            # Store report
            await self.db.llm_resonance_reports.insert_one(report.to_dict())
            
            logger.info(
                f"Resonance Report #{report_number:03d} generated: "
                f"Resonance={overall_resonance:.3f}, Stability={overall_stability:.3f}"
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating resonance report: {e}")
            raise
    
    def _generate_report_recommendations(
        self,
        resonance: float,
        stability: float,
        equilibrium: float,
        entropy: float,
        drift_analysis: Dict[str, Any],
        oscillation_analysis: Dict[str, Any],
        entropy_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for resonance report"""
        recommendations = []
        
        # Resonance recommendations
        if resonance >= self.target_metrics["resonance_index"]:
            recommendations.append("✅ Resonance index meeting target - system alignment excellent")
        else:
            recommendations.append(
                f"⚠️ Resonance index ({resonance:.3f}) below target - enhance cross-module synchronization"
            )
        
        # Stability recommendations
        if stability >= self.target_metrics["temporal_stability"]:
            recommendations.append("✅ Temporal stability meeting target - system stable over time")
        else:
            recommendations.append(
                f"⚠️ Temporal stability ({stability:.3f}) below target - reduce parameter change rates"
            )
        
        # Equilibrium recommendations
        target_eq = self.target_metrics["feedback_equilibrium"]
        tolerance = self.target_metrics["feedback_equilibrium_tolerance"]
        if abs(equilibrium - target_eq) <= tolerance:
            recommendations.append("✅ Feedback equilibrium within target range - balanced learning")
        else:
            if equilibrium < target_eq - tolerance:
                recommendations.append("⚠️ Feedback equilibrium too low - increase novelty weight")
            else:
                recommendations.append("⚠️ Feedback equilibrium too high - increase stability weight")
        
        # Entropy recommendations
        if self.target_metrics["entropy_balance_min"] <= entropy <= self.target_metrics["entropy_balance_max"]:
            recommendations.append("✅ Entropy balance in target range - optimal exploration/exploitation")
        else:
            if entropy < self.target_metrics["entropy_balance_min"]:
                recommendations.append("⚠️ Entropy too low - risk of stagnation, increase novelty")
            else:
                recommendations.append("⚠️ Entropy too high - risk of instability, increase stability focus")
        
        # Drift recommendations
        if drift_analysis["drift_violations"] > 0:
            recommendations.append(
                f"🚨 {drift_analysis['drift_violations']} drift violations detected - "
                "implement drift control mechanisms"
            )
        
        # Oscillation recommendations
        if oscillation_analysis["oscillation_violations"] > 0:
            recommendations.append(
                f"🚨 {oscillation_analysis['oscillation_violations']} oscillation violations detected - "
                "stabilize parameter updates"
            )
        
        return recommendations[:8]
    
    def _prioritize_interventions(
        self,
        resonance: float,
        stability: float,
        equilibrium: float,
        entropy: float,
        drift_analysis: Dict[str, Any],
        oscillation_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Prioritize interventions by urgency"""
        interventions = []
        
        # Priority 1: Critical violations
        if drift_analysis["drift_violations"] > 0:
            interventions.append({
                "priority": "critical",
                "intervention": "Drift Control",
                "description": f"{drift_analysis['drift_violations']} drift violations - apply AFR corrections",
                "urgency_score": 0.95
            })
        
        if oscillation_analysis["oscillation_violations"] > 0:
            interventions.append({
                "priority": "critical",
                "intervention": "Oscillation Damping",
                "description": f"{oscillation_analysis['oscillation_violations']} oscillation violations - reduce update frequency",
                "urgency_score": 0.90
            })
        
        # Priority 2: Target misses
        if resonance < self.target_metrics["resonance_index"]:
            interventions.append({
                "priority": "high",
                "intervention": "Resonance Enhancement",
                "description": f"Resonance at {resonance:.3f} - trigger cohesion sync",
                "urgency_score": 0.80
            })
        
        if stability < self.target_metrics["temporal_stability"]:
            interventions.append({
                "priority": "high",
                "intervention": "Stability Improvement",
                "description": f"Stability at {stability:.3f} - reduce parameter volatility",
                "urgency_score": 0.75
            })
        
        # Priority 3: Balance adjustments
        if abs(equilibrium - 0.50) > 0.05:
            interventions.append({
                "priority": "medium",
                "intervention": "Equilibrium Rebalancing",
                "description": f"Equilibrium at {equilibrium:.3f} - adjust feedback weights",
                "urgency_score": 0.60
            })
        
        if entropy < 0.45 or entropy > 0.55:
            interventions.append({
                "priority": "medium",
                "intervention": "Entropy Correction",
                "description": f"Entropy at {entropy:.3f} - rebalance novelty/stability",
                "urgency_score": 0.55
            })
        
        # Sort by urgency
        interventions.sort(key=lambda x: x["urgency_score"], reverse=True)
        
        return interventions[:5]
    
    async def _generate_health_summary(
        self,
        resonance: float,
        stability: float,
        equilibrium: float,
        entropy: float
    ) -> str:
        """Generate overall system health summary"""
        
        if self.llm_available:
            return await self._llm_generate_health_summary(resonance, stability, equilibrium, entropy)
        else:
            return self._mock_generate_health_summary(resonance, stability, equilibrium, entropy)
    
    async def _llm_generate_health_summary(
        self,
        resonance: float,
        stability: float,
        equilibrium: float,
        entropy: float
    ) -> str:
        """Use LLM to generate health summary [PROD]"""
        try:
            provider_config = self.llm_providers["primary"]
            
            chat = LlmChat(
                api_key=self.llm_key,
                session_id=f"resonance-health-{uuid.uuid4()}",
                system_message="You are a cognitive systems analyst evaluating the resonance and stability of an AI chess system."
            ).with_model(provider_config["provider"], provider_config["model"])
            
            prompt = f"""Generate a concise health summary (3-4 sentences) for the Cognitive Resonance Framework:

**Current Metrics:**
- Resonance Index: {resonance:.3f} (target: ≥{self.target_metrics['resonance_index']:.2f})
- Temporal Stability: {stability:.3f} (target: ≥{self.target_metrics['temporal_stability']:.2f})
- Feedback Equilibrium: {equilibrium:.3f} (target: {self.target_metrics['feedback_equilibrium']:.2f} ± {self.target_metrics['feedback_equilibrium_tolerance']:.2f})
- Entropy Balance: {entropy:.3f} (target: {self.target_metrics['entropy_balance_min']:.2f}-{self.target_metrics['entropy_balance_max']:.2f})

Assess:
1. Overall resonance and stability health
2. Alignment with target metrics
3. System balance and cohesion
4. Forward-looking recommendation

Keep it under 100 words and actionable."""

            user_message = UserMessage(text=prompt)
            response = await chat.send_message(user_message)
            
            summary = response.strip()
            logger.info("[PROD] LLM-generated resonance health summary")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating LLM health summary: {e}")
            return self._mock_generate_health_summary(resonance, stability, equilibrium, entropy)
    
    def _mock_generate_health_summary(
        self,
        resonance: float,
        stability: float,
        equilibrium: float,
        entropy: float
    ) -> str:
        """Generate mock health summary [MOCK]"""
        
        targets_met = sum([
            resonance >= self.target_metrics["resonance_index"],
            stability >= self.target_metrics["temporal_stability"],
            abs(equilibrium - 0.50) <= 0.05,
            0.45 <= entropy <= 0.55
        ])
        
        if targets_met == 4:
            status = "excellent resonance health with all target metrics met"
        elif targets_met >= 3:
            status = "good resonance health with most targets achieved"
        elif targets_met >= 2:
            status = "moderate resonance health requiring attention"
        else:
            status = "resonance health needs improvement"
        
        return (
            f"Cognitive Resonance Framework operating at {status}. "
            f"Resonance index at {resonance:.3f} {'meets' if resonance >= 0.90 else 'approaching'} target, "
            f"with temporal stability at {stability:.3f}. "
            f"Feedback equilibrium at {equilibrium:.3f} and entropy balance at {entropy:.3f} indicate "
            f"{'balanced' if 0.45 <= equilibrium <= 0.55 and 0.45 <= entropy <= 0.55 else 'adjustment needed in'} "
            f"system dynamics. "
            f"{'Continue current resonance approach with regular monitoring.' if targets_met >= 3 else 'Implement recommended balance corrections to enhance long-term stability.'}"
        )
    
    async def get_resonance_status(self) -> Dict[str, Any]:
        """Get current resonance system status"""
        try:
            # Get latest metrics
            latest_metrics = await self.db.llm_resonance_metrics.find_one(
                sort=[("timestamp", -1)]
            )
            
            if latest_metrics:
                # Remove MongoDB _id
                if '_id' in latest_metrics:
                    del latest_metrics['_id']
                
                status = {
                    "system_status": "operational",
                    "last_analysis": latest_metrics.get("timestamp"),
                    "resonance_index": latest_metrics.get("resonance_index"),
                    "temporal_stability": latest_metrics.get("temporal_stability"),
                    "feedback_equilibrium": latest_metrics.get("feedback_equilibrium"),
                    "entropy_balance": latest_metrics.get("entropy_balance"),
                    "resonance_health": latest_metrics.get("resonance_health"),
                    "stability_warnings": latest_metrics.get("stability_warnings", []),
                    "target_comparison": {
                        "resonance": {
                            "current": latest_metrics.get("resonance_index"),
                            "target": self.target_metrics["resonance_index"],
                            "status": "✅" if latest_metrics.get("resonance_index", 0) >= self.target_metrics["resonance_index"] else "⚠️"
                        },
                        "stability": {
                            "current": latest_metrics.get("temporal_stability"),
                            "target": self.target_metrics["temporal_stability"],
                            "status": "✅" if latest_metrics.get("temporal_stability", 0) >= self.target_metrics["temporal_stability"] else "⚠️"
                        },
                        "equilibrium": {
                            "current": latest_metrics.get("feedback_equilibrium"),
                            "target": f"{self.target_metrics['feedback_equilibrium']} ± {self.target_metrics['feedback_equilibrium_tolerance']}",
                            "status": "✅" if abs(latest_metrics.get("feedback_equilibrium", 0.50) - 0.50) <= 0.05 else "⚠️"
                        },
                        "entropy": {
                            "current": latest_metrics.get("entropy_balance"),
                            "target": f"{self.target_metrics['entropy_balance_min']}-{self.target_metrics['entropy_balance_max']}",
                            "status": "✅" if self.target_metrics["entropy_balance_min"] <= latest_metrics.get("entropy_balance", 0.50) <= self.target_metrics["entropy_balance_max"] else "⚠️"
                        }
                    }
                }
            else:
                status = {
                    "system_status": "initializing",
                    "message": "No resonance analysis executed yet"
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting resonance status: {e}")
            return {"system_status": "error", "error": str(e)}
