"""
Cohesion Core & Systemic Unification (Step 32)

This module implements the unified cognitive hub that synchronizes and harmonizes
all intelligence subsystems:
- Creativity (Step 29): Autonomous creative synthesis
- Self-Reflection (Step 30): Continuous learning loop
- Memory Fusion (Step 31): Long-term cognitive persistence

Features:
- Unified cognitive synchronization across all subsystems
- Adaptive weight balancing (novelty, stability, ethics)
- Inter-module messaging and coordination
- System health monitoring and auto-healing
- Comprehensive cohesion reporting with transparency
- Real-time alignment scoring and drift detection
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
class ModuleState:
    """State snapshot of a cognitive module"""
    module_name: str
    timestamp: str
    parameters: Dict[str, float]
    health_score: float  # 0-1
    ethical_alignment: float  # 0-1
    activity_level: float  # 0-1: How active the module is
    last_update: str
    status: str  # "operational", "degraded", "inactive"
    
    def to_dict(self):
        return asdict(self)


@dataclass
class CohesionMetrics:
    """Comprehensive system cohesion metrics"""
    cycle_id: str
    timestamp: str
    alignment_score: float  # 0-1: Overall system alignment
    system_health_index: float  # 0-1: Overall system health
    ethical_continuity: float  # 0-1: Ethical consistency across modules
    synchronization_latency: float  # seconds
    
    # Delta measurements
    creativity_reflection_delta: float  # Difference in parameters
    memory_ethics_delta: float  # Ethical alignment difference
    parameter_harmony_score: float  # 0-1: How aligned parameters are
    
    # Module-specific scores
    creativity_health: float
    reflection_health: float
    memory_health: float
    
    # Action tracking
    actions_taken: List[str]
    drift_detected: bool
    auto_healing_applied: bool
    cohesion_health: str  # "excellent", "good", "moderate", "needs_attention"
    recommendations: List[str]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class CohesionReport:
    """Comprehensive cohesion report"""
    report_id: str
    cycle_id: str
    timestamp: str
    
    # Module states
    module_states: Dict[str, ModuleState]
    
    # Metrics
    metrics: CohesionMetrics
    
    # Parameter comparison
    parameter_comparison: Dict[str, Any]
    
    # Health analysis
    health_analysis: str
    
    # Recommendations
    recommendations: List[str]
    
    # Actions taken
    actions_log: List[Dict[str, Any]]
    
    def to_dict(self):
        result = asdict(self)
        # Convert module_states to dict format
        result['module_states'] = {k: v.to_dict() if hasattr(v, 'to_dict') else v 
                                   for k, v in self.module_states.items()}
        return result


class CohesionController:
    """
    Cohesion Core & Systemic Unification Controller
    
    Central synchronization point that harmonizes creativity, reflection,
    and memory subsystems for coherent cognitive operation.
    """
    
    def __init__(self, db_client):
        self.db = db_client
        self.llm_key = os.environ.get('EMERGENT_LLM_KEY')
        
        if not self.llm_key:
            logger.warning("EMERGENT_LLM_KEY not found - cohesion will use mock mode")
            self.llm_available = False
        else:
            self.llm_available = True
            logger.info("Cohesion Controller initialized with multi-provider LLM support")
        
        # Target metrics (from problem statement)
        self.target_metrics = {
            "alignment_score": 0.90,
            "system_health_index": 0.88,
            "ethical_continuity": 0.92,
            "synchronization_latency": 2.0
        }
        
        # Acceptable parameter drift thresholds
        self.drift_thresholds = {
            "novelty_weight": 0.10,  # ±10% acceptable
            "stability_weight": 0.10,
            "ethical_threshold": 0.05,  # ±5% for ethics (more strict)
            "creativity_bias": 0.08,
            "risk_tolerance": 0.10
        }
        
        # Auto-healing parameters
        self.auto_heal_enabled = True
        self.max_correction_per_cycle = 0.05  # Maximum 5% correction per cycle
        
        # LLM providers (per problem statement)
        self.llm_providers = {
            "primary": {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
            "secondary": {"provider": "openai", "model": "gpt-4o-mini"},
            "fallback": {"provider": "google", "model": "gemini-2.0-flash-exp"}
        }
        
        logger.info(
            f"Cohesion Core initialized: "
            f"Targets: alignment={self.target_metrics['alignment_score']}, "
            f"health={self.target_metrics['system_health_index']}, "
            f"ethics={self.target_metrics['ethical_continuity']}"
        )
    
    async def trigger_cohesion_cycle(
        self,
        trigger: str = "post_memory_fusion",
        memory_fusion_id: Optional[str] = None,
        reflection_cycle_id: Optional[str] = None
    ) -> CohesionReport:
        """
        Trigger a complete cohesion cycle.
        
        Called automatically after memory fusion cycles to ensure all subsystems
        remain aligned and harmonized.
        
        Args:
            trigger: What triggered this cycle
            memory_fusion_id: ID of the memory fusion that triggered this
            reflection_cycle_id: ID of the reflection cycle (if available)
        
        Returns:
            Comprehensive cohesion report
        """
        try:
            cycle_id = str(uuid.uuid4())
            start_time = datetime.now()
            
            logger.info(f"Starting cohesion cycle {cycle_id} (trigger: {trigger})")
            
            # Step 1: Gather module states
            module_states = await self._gather_module_states()
            
            # Step 2: Synchronize modules
            sync_results = await self.synchronize_modules(module_states)
            
            # Step 3: Evaluate system alignment
            alignment_metrics = await self.evaluate_system_alignment(module_states)
            
            # Step 4: Balance cognitive weights
            balance_results = await self.balance_cognitive_weights(module_states, alignment_metrics)
            
            # Step 5: Detect and correct drift
            drift_results = await self._detect_and_correct_drift(module_states)
            
            # Step 6: Calculate comprehensive metrics
            elapsed = (datetime.now() - start_time).total_seconds()
            metrics = self._build_cohesion_metrics(
                cycle_id, module_states, alignment_metrics, 
                balance_results, drift_results, elapsed
            )
            
            # Step 7: Store metrics
            await self.db.llm_cohesion_metrics.insert_one(metrics.to_dict())
            
            # Step 8: Update system health
            await self._update_system_health(metrics)
            
            # Step 9: Generate recommendations
            recommendations = self._generate_recommendations(metrics, module_states)
            
            # Step 10: Generate health analysis
            if self.llm_available:
                health_analysis = await self._llm_generate_health_analysis(
                    module_states, metrics, recommendations
                )
            else:
                health_analysis = self._mock_generate_health_analysis(metrics)
            
            # Step 11: Build parameter comparison
            param_comparison = self._build_parameter_comparison(module_states)
            
            # Step 12: Build actions log
            actions_log = self._build_actions_log(sync_results, balance_results, drift_results)
            
            # Create cohesion report
            report = CohesionReport(
                report_id=str(uuid.uuid4()),
                cycle_id=cycle_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                module_states=module_states,
                metrics=metrics,
                parameter_comparison=param_comparison,
                health_analysis=health_analysis,
                recommendations=recommendations,
                actions_log=actions_log
            )
            
            # Store report
            await self.db.llm_cohesion_reports.insert_one(report.to_dict())
            
            logger.info(
                f"Cohesion cycle {cycle_id} complete: "
                f"Alignment={metrics.alignment_score:.2f}, "
                f"Health={metrics.system_health_index:.2f}, "
                f"Ethics={metrics.ethical_continuity:.2f}, "
                f"Latency={elapsed:.2f}s"
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error in cohesion cycle: {e}")
            raise
    
    async def _gather_module_states(self) -> Dict[str, ModuleState]:
        """Gather current state of all cognitive modules"""
        try:
            states = {}
            
            # Get Creativity module state (Step 29)
            try:
                creativity_metrics = await self.db.llm_creative_synthesis.find_one(
                    sort=[("timestamp", -1)]
                )
                creativity_state = await self._extract_creativity_state(creativity_metrics)
                states["creativity"] = creativity_state
            except Exception as e:
                logger.warning(f"Could not get creativity state: {e}")
                states["creativity"] = self._get_default_module_state("creativity")
            
            # Get Reflection module state (Step 30)
            try:
                reflection_cycle = await self.db.llm_reflection_log.find_one(
                    sort=[("timestamp", -1)]
                )
                reflection_params = await self.db.llm_learning_parameters.find_one(
                    sort=[("timestamp", -1)]
                )
                reflection_state = await self._extract_reflection_state(
                    reflection_cycle, reflection_params
                )
                states["reflection"] = reflection_state
            except Exception as e:
                logger.warning(f"Could not get reflection state: {e}")
                states["reflection"] = self._get_default_module_state("reflection")
            
            # Get Memory Fusion module state (Step 31)
            try:
                memory_health = await self.db.llm_memory_nodes.count_documents({
                    "decay_weight": {"$gte": 0.20}
                })
                memory_metrics = await self.db.llm_memory_trace.find_one(
                    {"trace_type": "fusion_cycle"},
                    sort=[("timestamp", -1)]
                )
                memory_state = await self._extract_memory_state(memory_health, memory_metrics)
                states["memory"] = memory_state
            except Exception as e:
                logger.warning(f"Could not get memory state: {e}")
                states["memory"] = self._get_default_module_state("memory")
            
            logger.info(f"Gathered states for {len(states)} modules")
            return states
            
        except Exception as e:
            logger.error(f"Error gathering module states: {e}")
            return {}
    
    async def _extract_creativity_state(self, latest_metrics: Optional[Dict]) -> ModuleState:
        """Extract state from creativity module"""
        if not latest_metrics:
            return self._get_default_module_state("creativity")
        
        # Get recent strategies to assess health
        total_strategies = await self.db.llm_creative_synthesis.count_documents({})
        approved_strategies = await self.db.llm_creative_synthesis.count_documents({
            "rejected": False
        })
        
        health_score = (approved_strategies / total_strategies) if total_strategies > 0 else 0.70
        
        # Get average ethical alignment
        recent_strategies = await self.db.llm_creative_synthesis.find({
            "rejected": False
        }).sort("timestamp", -1).limit(10).to_list(10)
        
        if recent_strategies:
            avg_ethical = np.mean([s.get("ethical_alignment", 0.9) for s in recent_strategies])
            avg_novelty = np.mean([s.get("novelty_score", 0.7) for s in recent_strategies])
            avg_stability = np.mean([s.get("stability_score", 0.65) for s in recent_strategies])
        else:
            avg_ethical = 0.90
            avg_novelty = 0.70
            avg_stability = 0.65
        
        return ModuleState(
            module_name="creativity",
            timestamp=datetime.now(timezone.utc).isoformat(),
            parameters={
                "novelty_threshold": 0.60,
                "stability_threshold": 0.50,
                "ethical_threshold": 0.75,
                "avg_novelty": avg_novelty,
                "avg_stability": avg_stability
            },
            health_score=round(health_score, 2),
            ethical_alignment=round(avg_ethical, 2),
            activity_level=min(1.0, total_strategies / 50.0) if total_strategies > 0 else 0.0,
            last_update=latest_metrics.get("timestamp", datetime.now(timezone.utc).isoformat()),
            status="operational" if health_score > 0.60 else "degraded"
        )
    
    async def _extract_reflection_state(
        self, 
        latest_cycle: Optional[Dict],
        latest_params: Optional[Dict]
    ) -> ModuleState:
        """Extract state from reflection module"""
        if not latest_cycle or not latest_params:
            return self._get_default_module_state("reflection")
        
        # Extract parameters
        params = {
            "novelty_weight": latest_params.get("novelty_weight", 0.60),
            "stability_weight": latest_params.get("stability_weight", 0.50),
            "ethical_threshold": latest_params.get("ethical_threshold", 0.75),
            "creativity_bias": latest_params.get("creativity_bias", 0.55),
            "risk_tolerance": latest_params.get("risk_tolerance", 0.50)
        }
        
        # Get health metrics
        learning_health = latest_cycle.get("learning_health_index", 0.75)
        ethical_status = latest_cycle.get("ethical_alignment_status", "good")
        
        ethical_score_map = {
            "excellent": 0.95,
            "good": 0.85,
            "needs_attention": 0.70,
            "critical": 0.50
        }
        ethical_score = ethical_score_map.get(ethical_status, 0.80)
        
        # Calculate activity level
        games_analyzed = latest_cycle.get("games_analyzed", 0)
        activity_level = min(1.0, games_analyzed / 5.0)
        
        return ModuleState(
            module_name="reflection",
            timestamp=datetime.now(timezone.utc).isoformat(),
            parameters=params,
            health_score=round(learning_health, 2),
            ethical_alignment=round(ethical_score, 2),
            activity_level=round(activity_level, 2),
            last_update=latest_cycle.get("timestamp", datetime.now(timezone.utc).isoformat()),
            status="operational" if learning_health > 0.60 else "degraded"
        )
    
    async def _extract_memory_state(
        self,
        active_nodes_count: int,
        latest_metrics: Optional[Dict]
    ) -> ModuleState:
        """Extract state from memory fusion module"""
        if not latest_metrics:
            return self._get_default_module_state("memory")
        
        # Get memory health from latest fusion
        fusion_metrics = latest_metrics.get("fusion_metrics", {})
        
        persistence_health = fusion_metrics.get("persistence_health", 0.80)
        ethical_continuity = fusion_metrics.get("ethical_continuity", 0.90)
        memory_retention = fusion_metrics.get("memory_retention_index", 0.85)
        
        # Calculate activity level based on active nodes
        activity_level = min(1.0, active_nodes_count / 20.0)
        
        return ModuleState(
            module_name="memory",
            timestamp=datetime.now(timezone.utc).isoformat(),
            parameters={
                "decay_lambda": 0.05,
                "retention_window": 30,
                "memory_retention": memory_retention,
                "fusion_efficiency": fusion_metrics.get("fusion_efficiency", 0.80)
            },
            health_score=round(persistence_health, 2),
            ethical_alignment=round(ethical_continuity, 2),
            activity_level=round(activity_level, 2),
            last_update=latest_metrics.get("timestamp", datetime.now(timezone.utc).isoformat()),
            status="operational" if persistence_health > 0.70 else "degraded"
        )
    
    def _get_default_module_state(self, module_name: str) -> ModuleState:
        """Get default state for a module (used when data unavailable)"""
        defaults = {
            "creativity": {
                "parameters": {
                    "novelty_threshold": 0.60,
                    "stability_threshold": 0.50,
                    "ethical_threshold": 0.75
                },
                "health_score": 0.75,
                "ethical_alignment": 0.88
            },
            "reflection": {
                "parameters": {
                    "novelty_weight": 0.60,
                    "stability_weight": 0.50,
                    "ethical_threshold": 0.75,
                    "creativity_bias": 0.55,
                    "risk_tolerance": 0.50
                },
                "health_score": 0.75,
                "ethical_alignment": 0.85
            },
            "memory": {
                "parameters": {
                    "decay_lambda": 0.05,
                    "retention_window": 30,
                    "memory_retention": 0.85
                },
                "health_score": 0.80,
                "ethical_alignment": 0.92
            }
        }
        
        default = defaults.get(module_name, {})
        
        return ModuleState(
            module_name=module_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            parameters=default.get("parameters", {}),
            health_score=default.get("health_score", 0.70),
            ethical_alignment=default.get("ethical_alignment", 0.85),
            activity_level=0.50,
            last_update=datetime.now(timezone.utc).isoformat(),
            status="inactive"
        )
    
    async def synchronize_modules(
        self,
        module_states: Dict[str, ModuleState]
    ) -> Dict[str, Any]:
        """
        Cross-check parameters across Steps 29-31 for coherence.
        
        Returns:
            Synchronization results with coherence scores
        """
        try:
            logger.info("Synchronizing modules for coherence...")
            
            if not module_states or len(module_states) < 2:
                return {"success": False, "message": "Insufficient module data"}
            
            # Extract common parameters across modules
            common_params = ["novelty_weight", "stability_weight", "ethical_threshold"]
            
            sync_results = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "modules_synced": list(module_states.keys()),
                "parameter_deltas": {},
                "coherence_score": 0.0,
                "issues_found": [],
                "corrections_applied": []
            }
            
            # Compare parameters across modules
            for param in common_params:
                values = []
                for module_name, state in module_states.items():
                    if param in state.parameters:
                        values.append(state.parameters[param])
                
                if len(values) >= 2:
                    param_delta = max(values) - min(values)
                    sync_results["parameter_deltas"][param] = round(param_delta, 3)
                    
                    # Flag if delta exceeds threshold
                    if param_delta > self.drift_thresholds.get(param, 0.10):
                        sync_results["issues_found"].append(
                            f"Parameter '{param}' drift detected: Δ={param_delta:.3f}"
                        )
            
            # Calculate coherence score (1.0 = perfect alignment, 0.0 = complete misalignment)
            if sync_results["parameter_deltas"]:
                avg_delta = np.mean(list(sync_results["parameter_deltas"].values()))
                # Inverse relationship: smaller delta = higher coherence
                coherence_score = max(0.0, 1.0 - (avg_delta / 0.15))
                sync_results["coherence_score"] = round(coherence_score, 2)
            else:
                sync_results["coherence_score"] = 0.90  # Default if no comparable params
            
            logger.info(
                f"Module synchronization complete: "
                f"coherence={sync_results['coherence_score']:.2f}, "
                f"issues={len(sync_results['issues_found'])}"
            )
            
            return sync_results
            
        except Exception as e:
            logger.error(f"Error synchronizing modules: {e}")
            return {"success": False, "error": str(e)}
    
    async def evaluate_system_alignment(
        self,
        module_states: Dict[str, ModuleState]
    ) -> Dict[str, Any]:
        """
        Compute unified health index across creativity, reflection, memory, and ethics.
        
        Returns:
            Comprehensive alignment metrics
        """
        try:
            logger.info("Evaluating system alignment...")
            
            if not module_states:
                return {"alignment_score": 0.0, "error": "No module states available"}
            
            # Component 1: Health alignment (30%)
            health_scores = [state.health_score for state in module_states.values()]
            avg_health = np.mean(health_scores)
            health_std = np.std(health_scores)
            health_alignment = avg_health * (1.0 - min(0.3, health_std))  # Penalize variance
            
            # Component 2: Ethical alignment (35%)
            ethical_scores = [state.ethical_alignment for state in module_states.values()]
            avg_ethical = np.mean(ethical_scores)
            ethical_std = np.std(ethical_scores)
            ethical_alignment = avg_ethical * (1.0 - min(0.2, ethical_std))
            
            # Component 3: Activity alignment (20%)
            activity_levels = [state.activity_level for state in module_states.values()]
            avg_activity = np.mean(activity_levels)
            activity_std = np.std(activity_levels)
            activity_alignment = avg_activity * (1.0 - min(0.4, activity_std))
            
            # Component 4: Status consistency (15%)
            operational_count = sum(1 for state in module_states.values() if state.status == "operational")
            status_score = operational_count / len(module_states)
            
            # Calculate overall alignment score
            alignment_score = (
                health_alignment * 0.30 +
                ethical_alignment * 0.35 +
                activity_alignment * 0.20 +
                status_score * 0.15
            )
            
            alignment_metrics = {
                "alignment_score": round(alignment_score, 2),
                "health_alignment": round(health_alignment, 2),
                "ethical_alignment": round(ethical_alignment, 2),
                "activity_alignment": round(activity_alignment, 2),
                "status_consistency": round(status_score, 2),
                "avg_health": round(avg_health, 2),
                "avg_ethical": round(avg_ethical, 2),
                "avg_activity": round(avg_activity, 2),
                "operational_modules": operational_count,
                "total_modules": len(module_states)
            }
            
            logger.info(
                f"System alignment evaluated: "
                f"score={alignment_score:.2f}, "
                f"health={avg_health:.2f}, "
                f"ethics={avg_ethical:.2f}"
            )
            
            return alignment_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating system alignment: {e}")
            return {"alignment_score": 0.0, "error": str(e)}
    
    async def balance_cognitive_weights(
        self,
        module_states: Dict[str, ModuleState],
        alignment_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Dynamically harmonize novelty, stability, and ethics parameters across modules.
        
        Returns:
            Balance adjustments and recommendations
        """
        try:
            logger.info("Balancing cognitive weights...")
            
            balance_results = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "adjustments": {},
                "recommendations": [],
                "balance_score": 0.0
            }
            
            # Extract current weights from reflection module (authoritative source)
            reflection_state = module_states.get("reflection")
            if not reflection_state:
                logger.warning("Reflection module state not available for balancing")
                return balance_results
            
            current_novelty = reflection_state.parameters.get("novelty_weight", 0.60)
            current_stability = reflection_state.parameters.get("stability_weight", 0.50)
            current_ethical = reflection_state.parameters.get("ethical_threshold", 0.75)
            
            # Check if weights need adjustment based on alignment
            alignment_score = alignment_metrics.get("alignment_score", 0.80)
            avg_ethical = alignment_metrics.get("avg_ethical", 0.85)
            
            # Rule 1: If alignment is low, increase stability
            if alignment_score < 0.80:
                stability_adjustment = +0.03
                novelty_adjustment = -0.02
                balance_results["adjustments"]["stability_weight"] = stability_adjustment
                balance_results["adjustments"]["novelty_weight"] = novelty_adjustment
                balance_results["recommendations"].append(
                    f"Low alignment ({alignment_score:.2f}) detected - recommending stability increase"
                )
            
            # Rule 2: If ethical scores are declining, tighten ethical threshold
            if avg_ethical < 0.85:
                ethical_adjustment = +0.02
                balance_results["adjustments"]["ethical_threshold"] = ethical_adjustment
                balance_results["recommendations"].append(
                    f"Ethical scores below target ({avg_ethical:.2f}) - recommending threshold increase"
                )
            
            # Rule 3: If alignment is excellent, can increase novelty
            if alignment_score >= 0.90 and avg_ethical >= 0.90:
                novelty_adjustment = +0.02
                balance_results["adjustments"]["novelty_weight"] = novelty_adjustment
                balance_results["recommendations"].append(
                    f"Excellent alignment ({alignment_score:.2f}) - can explore more novelty"
                )
            
            # Calculate balance score
            # Perfect balance: novelty + stability ≈ 1.1, ethical > 0.75
            weight_sum = current_novelty + current_stability
            target_sum = 1.1
            sum_deviation = abs(weight_sum - target_sum)
            ethical_deviation = max(0, 0.75 - current_ethical)
            
            balance_score = max(0.0, 1.0 - sum_deviation - ethical_deviation)
            balance_results["balance_score"] = round(balance_score, 2)
            
            # Add current state
            balance_results["current_weights"] = {
                "novelty_weight": current_novelty,
                "stability_weight": current_stability,
                "ethical_threshold": current_ethical
            }
            
            if not balance_results["adjustments"]:
                balance_results["recommendations"].append(
                    "Cognitive weights are well-balanced - no adjustments needed"
                )
            
            logger.info(
                f"Cognitive weight balancing complete: "
                f"balance_score={balance_score:.2f}, "
                f"adjustments={len(balance_results['adjustments'])}"
            )
            
            return balance_results
            
        except Exception as e:
            logger.error(f"Error balancing cognitive weights: {e}")
            return {"error": str(e)}
    
    async def _detect_and_correct_drift(
        self,
        module_states: Dict[str, ModuleState]
    ) -> Dict[str, Any]:
        """
        Detect parameter drift and apply minor auto-healing corrections.
        
        Returns:
            Drift detection results and corrections applied
        """
        try:
            drift_results = {
                "drift_detected": False,
                "drift_parameters": [],
                "corrections_applied": [],
                "auto_heal_enabled": self.auto_heal_enabled
            }
            
            # Compare creativity and reflection parameters
            creativity_state = module_states.get("creativity")
            reflection_state = module_states.get("reflection")
            
            if not creativity_state or not reflection_state:
                return drift_results
            
            # Check novelty drift (creativity avg_novelty vs reflection novelty_weight)
            creativity_novelty = creativity_state.parameters.get("avg_novelty", 0.70)
            reflection_novelty = reflection_state.parameters.get("novelty_weight", 0.60)
            novelty_drift = abs(creativity_novelty - reflection_novelty)
            
            if novelty_drift > self.drift_thresholds.get("novelty_weight", 0.10):
                drift_results["drift_detected"] = True
                drift_results["drift_parameters"].append({
                    "parameter": "novelty",
                    "drift_amount": round(novelty_drift, 3),
                    "creativity_value": creativity_novelty,
                    "reflection_value": reflection_novelty
                })
                
                if self.auto_heal_enabled:
                    # Apply small correction (move both toward average)
                    correction = min(self.max_correction_per_cycle, novelty_drift * 0.3)
                    drift_results["corrections_applied"].append({
                        "parameter": "novelty_weight",
                        "correction": f"±{correction:.3f}",
                        "action": "advisory"  # Advisory mode only
                    })
            
            # Check stability drift
            creativity_stability = creativity_state.parameters.get("avg_stability", 0.65)
            reflection_stability = reflection_state.parameters.get("stability_weight", 0.50)
            stability_drift = abs(creativity_stability - reflection_stability)
            
            if stability_drift > self.drift_thresholds.get("stability_weight", 0.10):
                drift_results["drift_detected"] = True
                drift_results["drift_parameters"].append({
                    "parameter": "stability",
                    "drift_amount": round(stability_drift, 3),
                    "creativity_value": creativity_stability,
                    "reflection_value": reflection_stability
                })
                
                if self.auto_heal_enabled:
                    correction = min(self.max_correction_per_cycle, stability_drift * 0.3)
                    drift_results["corrections_applied"].append({
                        "parameter": "stability_weight",
                        "correction": f"±{correction:.3f}",
                        "action": "advisory"
                    })
            
            # Check ethical drift across all modules
            ethical_scores = [state.ethical_alignment for state in module_states.values()]
            ethical_std = np.std(ethical_scores)
            
            if ethical_std > 0.08:  # More than 8% variance in ethics
                drift_results["drift_detected"] = True
                drift_results["drift_parameters"].append({
                    "parameter": "ethical_alignment",
                    "drift_amount": round(ethical_std, 3),
                    "variance": round(ethical_std, 3),
                    "min_value": round(min(ethical_scores), 2),
                    "max_value": round(max(ethical_scores), 2)
                })
                
                if self.auto_heal_enabled:
                    drift_results["corrections_applied"].append({
                        "parameter": "ethical_threshold",
                        "correction": "+0.02",
                        "action": "advisory - tighten ethical standards"
                    })
            
            logger.info(
                f"Drift detection complete: "
                f"drift={drift_results['drift_detected']}, "
                f"parameters={len(drift_results['drift_parameters'])}"
            )
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Error detecting drift: {e}")
            return {"drift_detected": False, "error": str(e)}
    
    def _build_cohesion_metrics(
        self,
        cycle_id: str,
        module_states: Dict[str, ModuleState],
        alignment_metrics: Dict[str, Any],
        balance_results: Dict[str, Any],
        drift_results: Dict[str, Any],
        elapsed_time: float
    ) -> CohesionMetrics:
        """Build comprehensive cohesion metrics"""
        
        # Extract key metrics
        alignment_score = alignment_metrics.get("alignment_score", 0.80)
        avg_health = alignment_metrics.get("avg_health", 0.75)
        avg_ethical = alignment_metrics.get("avg_ethical", 0.85)
        
        # Calculate deltas
        creativity_state = module_states.get("creativity")
        reflection_state = module_states.get("reflection")
        memory_state = module_states.get("memory")
        
        if creativity_state and reflection_state:
            creativity_reflection_delta = abs(
                creativity_state.health_score - reflection_state.health_score
            )
        else:
            creativity_reflection_delta = 0.0
        
        if memory_state:
            memory_ethics_delta = abs(memory_state.ethical_alignment - avg_ethical)
        else:
            memory_ethics_delta = 0.0
        
        # Parameter harmony score
        param_harmony = balance_results.get("balance_score", 0.85)
        
        # Module health scores
        creativity_health = creativity_state.health_score if creativity_state else 0.75
        reflection_health = reflection_state.health_score if reflection_state else 0.75
        memory_health = memory_state.health_score if memory_state else 0.80
        
        # Calculate system health index (weighted combination)
        system_health = (
            avg_health * 0.40 +
            param_harmony * 0.30 +
            alignment_score * 0.30
        )
        
        # Actions taken
        actions_taken = []
        if drift_results.get("corrections_applied"):
            for correction in drift_results["corrections_applied"]:
                actions_taken.append(
                    f"adjust_{correction['parameter']} {correction['correction']}"
                )
        if balance_results.get("adjustments"):
            for param, adj in balance_results["adjustments"].items():
                actions_taken.append(f"adjust_{param} {adj:+.2f}")
        
        # Determine cohesion health status
        if system_health >= 0.90 and avg_ethical >= 0.92:
            cohesion_health = "excellent"
        elif system_health >= 0.80 and avg_ethical >= 0.85:
            cohesion_health = "good"
        elif system_health >= 0.65:
            cohesion_health = "moderate"
        else:
            cohesion_health = "needs_attention"
        
        # Build recommendations
        recommendations = balance_results.get("recommendations", [])
        if drift_results.get("drift_detected"):
            recommendations.append("Monitor parameter drift in upcoming cycles")
        if system_health < self.target_metrics["system_health_index"]:
            recommendations.append(
                f"System health ({system_health:.2f}) below target "
                f"({self.target_metrics['system_health_index']:.2f})"
            )
        
        metrics = CohesionMetrics(
            cycle_id=cycle_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            alignment_score=round(alignment_score, 2),
            system_health_index=round(system_health, 2),
            ethical_continuity=round(avg_ethical, 2),
            synchronization_latency=round(elapsed_time, 2),
            creativity_reflection_delta=round(creativity_reflection_delta, 2),
            memory_ethics_delta=round(memory_ethics_delta, 2),
            parameter_harmony_score=round(param_harmony, 2),
            creativity_health=round(creativity_health, 2),
            reflection_health=round(reflection_health, 2),
            memory_health=round(memory_health, 2),
            actions_taken=actions_taken,
            drift_detected=drift_results.get("drift_detected", False),
            auto_healing_applied=len(drift_results.get("corrections_applied", [])) > 0,
            cohesion_health=cohesion_health,
            recommendations=recommendations[:5]  # Top 5 recommendations
        )
        
        return metrics
    
    async def _update_system_health(self, metrics: CohesionMetrics):
        """Update system health history"""
        try:
            health_doc = {
                "cycle_id": metrics.cycle_id,
                "timestamp": metrics.timestamp,
                "system_health_index": metrics.system_health_index,
                "alignment_score": metrics.alignment_score,
                "ethical_continuity": metrics.ethical_continuity,
                "cohesion_health": metrics.cohesion_health,
                "module_health": {
                    "creativity": metrics.creativity_health,
                    "reflection": metrics.reflection_health,
                    "memory": metrics.memory_health
                }
            }
            
            await self.db.llm_system_health.insert_one(health_doc)
            logger.info(f"System health updated: {metrics.cohesion_health}")
            
        except Exception as e:
            logger.error(f"Error updating system health: {e}")
    
    def _generate_recommendations(
        self,
        metrics: CohesionMetrics,
        module_states: Dict[str, ModuleState]
    ) -> List[str]:
        """Generate actionable recommendations based on cohesion metrics"""
        
        recommendations = []
        
        # Alignment-based recommendations
        if metrics.alignment_score >= 0.90:
            recommendations.append(
                "✅ Excellent alignment - system operating in optimal coherence"
            )
        elif metrics.alignment_score >= 0.80:
            recommendations.append(
                "✅ Good alignment - minor refinements may enhance coherence"
            )
        elif metrics.alignment_score >= 0.70:
            recommendations.append(
                "⚠️ Moderate alignment - review parameter synchronization"
            )
        else:
            recommendations.append(
                "🚨 Low alignment - immediate parameter review recommended"
            )
        
        # Health-based recommendations
        if metrics.system_health_index < self.target_metrics["system_health_index"]:
            recommendations.append(
                f"System health ({metrics.system_health_index:.2f}) below target - "
                f"investigate module-specific issues"
            )
        
        # Ethics-based recommendations
        if metrics.ethical_continuity < self.target_metrics["ethical_continuity"]:
            recommendations.append(
                f"Ethical continuity ({metrics.ethical_continuity:.2f}) below target - "
                f"tighten ethical constraints"
            )
        elif metrics.ethical_continuity >= 0.92:
            recommendations.append(
                "✅ Outstanding ethical continuity maintained across all modules"
            )
        
        # Latency-based recommendations
        if metrics.synchronization_latency > self.target_metrics["synchronization_latency"]:
            recommendations.append(
                f"⚠️ Synchronization latency ({metrics.synchronization_latency:.2f}s) "
                f"exceeds target - optimize cohesion cycle"
            )
        
        # Drift-based recommendations
        if metrics.drift_detected:
            recommendations.append(
                "⚠️ Parameter drift detected - continue monitoring for stability"
            )
        
        # Parameter harmony recommendations
        if metrics.parameter_harmony_score < 0.80:
            recommendations.append(
                f"Parameter harmony ({metrics.parameter_harmony_score:.2f}) needs improvement - "
                f"balance novelty and stability weights"
            )
        
        # Module-specific recommendations
        if metrics.creativity_health < 0.70:
            recommendations.append(
                "🚨 Creativity module health low - review strategy generation quality"
            )
        if metrics.reflection_health < 0.70:
            recommendations.append(
                "🚨 Reflection module health low - increase reflection cycle frequency"
            )
        if metrics.memory_health < 0.75:
            recommendations.append(
                "🚨 Memory module health low - check memory retention and decay rates"
            )
        
        # Module activity recommendations
        for module_name, state in module_states.items():
            if state.activity_level < 0.30:
                recommendations.append(
                    f"📊 {module_name.capitalize()} module low activity - "
                    f"ensure module is actively engaged"
                )
        
        # Status recommendations
        degraded_modules = [name for name, state in module_states.items() 
                          if state.status == "degraded"]
        if degraded_modules:
            recommendations.append(
                f"⚠️ Degraded modules: {', '.join(degraded_modules)} - investigate root causes"
            )
        
        inactive_modules = [name for name, state in module_states.items() 
                           if state.status == "inactive"]
        if inactive_modules:
            recommendations.append(
                f"🚨 Inactive modules: {', '.join(inactive_modules)} - reactivate or troubleshoot"
            )
        
        # Auto-healing recommendations
        if metrics.auto_healing_applied:
            recommendations.append(
                "🔧 Auto-healing corrections applied - monitor for effectiveness"
            )
        
        # Overall status
        if metrics.cohesion_health == "excellent":
            recommendations.append(
                "🌟 System cohesion excellent - continue current operational parameters"
            )
        elif metrics.cohesion_health == "needs_attention":
            recommendations.append(
                "🚨 System cohesion needs attention - prioritize parameter alignment"
            )
        
        return recommendations[:8]  # Return top 8 most important
    
    async def _llm_generate_health_analysis(
        self,
        module_states: Dict[str, ModuleState],
        metrics: CohesionMetrics,
        recommendations: List[str]
    ) -> str:
        """Use LLM to generate comprehensive health analysis [PROD]"""
        
        try:
            provider_config = self.llm_providers["primary"]
            
            chat = LlmChat(
                api_key=self.llm_key,
                session_id=f"cohesion-analysis-{uuid.uuid4()}",
                system_message="You are a cognitive systems analyst for AlphaZero AI, evaluating the harmony and health of interconnected AI subsystems."
            ).with_model(provider_config["provider"], provider_config["model"])
            
            # Build module summary
            module_summary = ""
            for name, state in module_states.items():
                module_summary += f"\n- **{name.capitalize()}**: Health={state.health_score:.2f}, Ethics={state.ethical_alignment:.2f}, Status={state.status}"
            
            prompt = f"""Analyze the cohesion and health of the AlphaZero Chess AI cognitive system:

**Overall Metrics:**
- Alignment Score: {metrics.alignment_score:.2f} (target: ≥{self.target_metrics['alignment_score']:.2f})
- System Health: {metrics.system_health_index:.2f} (target: ≥{self.target_metrics['system_health_index']:.2f})
- Ethical Continuity: {metrics.ethical_continuity:.2f} (target: ≥{self.target_metrics['ethical_continuity']:.2f})
- Cohesion Status: {metrics.cohesion_health}

**Module States:**{module_summary}

**Parameter Harmony:** {metrics.parameter_harmony_score:.2f}
**Drift Detected:** {metrics.drift_detected}
**Actions Taken:** {len(metrics.actions_taken)}

Provide a concise health analysis (3-4 sentences) covering:
1. Overall system cohesion status
2. Key strengths and any concerns
3. Module synchronization quality
4. Forward-looking insight for maintaining harmony

Keep it under 100 words and actionable."""

            user_message = UserMessage(text=prompt)
            response = await chat.send_message(user_message)
            
            analysis = response.strip()
            logger.info("[PROD] LLM-generated cohesion health analysis")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating LLM health analysis: {e}")
            return self._mock_generate_health_analysis(metrics)
    
    def _mock_generate_health_analysis(self, metrics: CohesionMetrics) -> str:
        """Generate mock health analysis [MOCK]"""
        
        if metrics.cohesion_health == "excellent":
            return (
                f"System cohesion operating at excellent levels with alignment score of {metrics.alignment_score:.2f} "
                f"and system health at {metrics.system_health_index:.2f}. All subsystems (creativity, reflection, memory) "
                f"are well-synchronized with ethical continuity maintained at {metrics.ethical_continuity:.2f}. "
                f"Continue current operational parameters with regular monitoring. "
                f"Parameter harmony is strong ({metrics.parameter_harmony_score:.2f}), ensuring coherent cognitive operation."
            )
        elif metrics.cohesion_health == "good":
            return (
                f"System cohesion is good with alignment at {metrics.alignment_score:.2f} and health at {metrics.system_health_index:.2f}. "
                f"Minor parameter drift detected ({metrics.creativity_reflection_delta:.2f}) between creativity and reflection modules. "
                f"Ethical standards remain solid at {metrics.ethical_continuity:.2f}. "
                f"Recommended actions have been identified to enhance inter-module synchronization and maintain cognitive harmony."
            )
        elif metrics.cohesion_health == "moderate":
            return (
                f"System cohesion is moderate with alignment at {metrics.alignment_score:.2f}. "
                f"Parameter drift of {metrics.creativity_reflection_delta:.2f} detected requiring attention. "
                f"System health ({metrics.system_health_index:.2f}) below target, suggesting need for parameter rebalancing. "
                f"Ethical continuity at {metrics.ethical_continuity:.2f} remains acceptable. "
                f"Implement recommended parameter adjustments to restore optimal cohesion."
            )
        else:  # needs_attention
            return (
                f"System cohesion needs attention with alignment score at {metrics.alignment_score:.2f} below optimal. "
                f"Significant drift detected across modules with system health at {metrics.system_health_index:.2f}. "
                f"Immediate review of creativity-reflection synchronization recommended (Δ={metrics.creativity_reflection_delta:.2f}). "
                f"Ethical continuity at {metrics.ethical_continuity:.2f}. "
                f"Priority actions: parameter rebalancing, module synchronization, and increased monitoring frequency."
            )
    
    def _build_parameter_comparison(
        self,
        module_states: Dict[str, ModuleState]
    ) -> Dict[str, Any]:
        """Build parameter comparison across modules"""
        
        comparison = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "modules": {},
            "parameter_alignment": {}
        }
        
        # Extract parameters from each module
        for module_name, state in module_states.items():
            comparison["modules"][module_name] = {
                "parameters": state.parameters,
                "health": state.health_score,
                "ethics": state.ethical_alignment,
                "status": state.status
            }
        
        # Compare common parameters
        common_params = ["novelty_weight", "stability_weight", "ethical_threshold"]
        for param in common_params:
            values = []
            modules_with_param = []
            
            for module_name, state in module_states.items():
                # Check for direct match or related parameter
                param_value = None
                if param in state.parameters:
                    param_value = state.parameters[param]
                elif "avg_" + param.split("_")[0] in state.parameters:
                    param_value = state.parameters["avg_" + param.split("_")[0]]
                
                if param_value is not None:
                    values.append(param_value)
                    modules_with_param.append(module_name)
            
            if values:
                comparison["parameter_alignment"][param] = {
                    "values": {modules_with_param[i]: values[i] for i in range(len(values))},
                    "min": round(min(values), 3),
                    "max": round(max(values), 3),
                    "avg": round(np.mean(values), 3),
                    "delta": round(max(values) - min(values), 3),
                    "aligned": (max(values) - min(values)) < 0.10
                }
        
        return comparison
    
    def _build_actions_log(
        self,
        sync_results: Dict[str, Any],
        balance_results: Dict[str, Any],
        drift_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build comprehensive actions log"""
        
        actions_log = []
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Log synchronization actions
        if sync_results.get("issues_found"):
            for issue in sync_results["issues_found"]:
                actions_log.append({
                    "timestamp": timestamp,
                    "action_type": "synchronization",
                    "description": issue,
                    "status": "detected"
                })
        
        # Log balance adjustments
        if balance_results.get("adjustments"):
            for param, adjustment in balance_results["adjustments"].items():
                actions_log.append({
                    "timestamp": timestamp,
                    "action_type": "balance_adjustment",
                    "parameter": param,
                    "adjustment": adjustment,
                    "status": "recommended"
                })
        
        # Log drift corrections
        if drift_results.get("corrections_applied"):
            for correction in drift_results["corrections_applied"]:
                actions_log.append({
                    "timestamp": timestamp,
                    "action_type": "drift_correction",
                    "parameter": correction["parameter"],
                    "correction": correction["correction"],
                    "status": correction["action"]
                })
        
        # Log if no actions needed
        if not actions_log:
            actions_log.append({
                "timestamp": timestamp,
                "action_type": "monitoring",
                "description": "System cohesion within acceptable parameters",
                "status": "stable"
            })
        
        return actions_log
    
    async def get_cohesion_status(self) -> Dict[str, Any]:
        """Get current cohesion system status"""
        try:
            # Get latest cohesion report
            latest_report = await self.db.llm_cohesion_reports.find_one(
                sort=[("timestamp", -1)]
            )
            
            if latest_report:
                metrics = latest_report.get("metrics", {})
                status = {
                    "last_cycle": latest_report.get("timestamp"),
                    "cycle_id": latest_report.get("cycle_id"),
                    "alignment_score": metrics.get("alignment_score", 0),
                    "system_health_index": metrics.get("system_health_index", 0),
                    "ethical_continuity": metrics.get("ethical_continuity", 0),
                    "cohesion_health": metrics.get("cohesion_health", "unknown"),
                    "drift_detected": metrics.get("drift_detected", False),
                    "auto_healing_active": metrics.get("auto_healing_applied", False),
                    "system_status": "operational"
                }
            else:
                status = {
                    "last_cycle": None,
                    "system_status": "initializing",
                    "message": "No cohesion cycles executed yet"
                }
            
            # Add target comparison
            if latest_report:
                status["target_comparison"] = {
                    "alignment": {
                        "current": metrics.get("alignment_score", 0),
                        "target": self.target_metrics["alignment_score"],
                        "status": "✅" if metrics.get("alignment_score", 0) >= self.target_metrics["alignment_score"] else "⚠️"
                    },
                    "health": {
                        "current": metrics.get("system_health_index", 0),
                        "target": self.target_metrics["system_health_index"],
                        "status": "✅" if metrics.get("system_health_index", 0) >= self.target_metrics["system_health_index"] else "⚠️"
                    },
                    "ethics": {
                        "current": metrics.get("ethical_continuity", 0),
                        "target": self.target_metrics["ethical_continuity"],
                        "status": "✅" if metrics.get("ethical_continuity", 0) >= self.target_metrics["ethical_continuity"] else "⚠️"
                    }
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting cohesion status: {e}")
            return {
                "system_status": "error",
                "error": str(e)
            }
    
    async def get_cohesion_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get cohesion cycle history"""
        try:
            reports = await self.db.llm_cohesion_reports.find().sort(
                "timestamp", -1
            ).limit(limit).to_list(limit)
            
            return reports
            
        except Exception as e:
            logger.error(f"Error getting cohesion history: {e}")
            return []
    
    async def generate_cohesion_report(self, cycle_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive cohesion report"""
        try:
            if cycle_id:
                # Get specific cycle report
                report = await self.db.llm_cohesion_reports.find_one({"cycle_id": cycle_id})
            else:
                # Get latest report
                report = await self.db.llm_cohesion_reports.find_one(
                    sort=[("timestamp", -1)]
                )
            
            if not report:
                return {
                    "success": False,
                    "message": "No cohesion report found"
                }
            
            return {
                "success": True,
                "report": report
            }
            
        except Exception as e:
            logger.error(f"Error generating cohesion report: {e}")
            return {
                "success": False,
                "error": str(e)
            }
