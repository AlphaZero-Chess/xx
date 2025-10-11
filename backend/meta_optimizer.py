"""
Meta-Reasoning Controller for Autonomous Self-Optimization (Step 24)

This module implements emergent meta-reasoning capabilities that allow the system
to autonomously adjust its own reasoning parameters, trust thresholds, and 
synthesis weighting based on performance feedback and emergent behavior.
"""

import logging
import asyncio
import uuid
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System-level performance metrics"""
    timestamp: str
    subsystem: str  # "trust", "memory", "arbitration", "synthesis", "llm"
    alignment_pct: float  # 0-100
    trust_variance: float  # 0-1
    consensus_stability: float  # 0-1
    response_accuracy: float  # 0-100
    win_rate: float  # 0-1
    latency_ms: float
    
    def to_dict(self):
        return asdict(self)


@dataclass
class MetaAdjustment:
    """Proposed meta-level adjustment"""
    adjustment_id: str
    parameter_name: str
    current_value: float
    proposed_value: float
    change_percent: float
    reason: str
    confidence: float  # 0-1
    is_critical: bool  # If True, requires manual approval
    
    def to_dict(self):
        return asdict(self)


@dataclass
class OptimizationCycle:
    """Complete optimization cycle result"""
    cycle_id: str
    timestamp: str
    trigger: str  # "scheduled", "manual", "performance_threshold"
    adjustments: List[MetaAdjustment]
    simulation_results: Dict[str, Any]
    applied: bool
    approval_required: bool
    performance_delta: Dict[str, float]  # {metric: change}
    reflection_summary: str
    system_health_score: float  # 0-100
    
    def to_dict(self):
        result = asdict(self)
        result['adjustments'] = [adj.to_dict() if hasattr(adj, 'to_dict') else adj 
                                 for adj in result['adjustments']]
        return result


class MetaReasoningController:
    """
    Autonomous meta-reasoning controller that continuously evaluates system
    performance and derives optimal parameter adjustments.
    """
    
    def __init__(self, db_client):
        self.db = db_client
        self.optimization_interval = 3600  # 1 hour in seconds
        self.running = False
        self.last_optimization = None
        
        # Parameter bounds (non-critical parameters only)
        self.parameter_bounds = {
            "trust_threshold": (0.5, 0.95),
            "arbitration_threshold": (0.6, 0.95),
            "synthesis_weight_llm": (0.3, 0.7),
            "synthesis_weight_mcts": (0.3, 0.7),
            "exploration_ratio": (0.1, 0.5),
            "prompt_depth": (1, 10),
            "max_response_time": (5.0, 20.0)
        }
        
        # Critical parameters that require manual approval
        self.critical_parameters = {
            "learning_rate", "num_simulations", "c_puct", "temperature"
        }
    
    async def analyze_system_performance(self, lookback_hours: int = 24) -> Dict[str, SystemMetrics]:
        """
        Analyze system performance across all subsystems over specified timeframe.
        
        Returns:
            Dictionary mapping subsystem names to their current metrics
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
            
            metrics = {}
            
            # 1. Analyze LLM Performance
            llm_metrics = await self._analyze_llm_subsystem(cutoff_time)
            metrics["llm"] = llm_metrics
            
            # 2. Analyze Trust/Consensus System
            trust_metrics = await self._analyze_trust_subsystem(cutoff_time)
            metrics["trust"] = trust_metrics
            
            # 3. Analyze Memory/Distillation System
            memory_metrics = await self._analyze_memory_subsystem(cutoff_time)
            metrics["memory"] = memory_metrics
            
            # 4. Analyze Synthesis System
            synthesis_metrics = await self._analyze_synthesis_subsystem(cutoff_time)
            metrics["synthesis"] = synthesis_metrics
            
            # 5. Analyze Training/Evaluation System
            training_metrics = await self._analyze_training_subsystem(cutoff_time)
            metrics["training"] = training_metrics
            
            logger.info(f"System performance analysis complete: {len(metrics)} subsystems")
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing system performance: {e}")
            return {}
    
    async def _analyze_llm_subsystem(self, cutoff_time: datetime) -> SystemMetrics:
        """Analyze LLM performance metrics"""
        try:
            # Get recent LLM performance data
            llm_performance = await self.db.llm_performance.find(
                {"timestamp": {"$gte": cutoff_time.isoformat()}}
            ).to_list(1000)
            
            llm_feedback = await self.db.llm_feedback.find(
                {"timestamp": {"$gte": cutoff_time.isoformat()}}
            ).to_list(1000)
            
            if llm_performance:
                avg_latency = np.mean([p.get("response_time", 0) * 1000 for p in llm_performance])
                success_rate = np.mean([1 if p.get("success") else 0 for p in llm_performance])
            else:
                avg_latency = 0
                success_rate = 1.0
            
            if llm_feedback:
                accuracy = np.mean([f.get("accuracy_score", 0) / 5.0 * 100 for f in llm_feedback])
                variance = np.std([f.get("accuracy_score", 0) for f in llm_feedback])
            else:
                accuracy = 80.0
                variance = 0.1
            
            return SystemMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                subsystem="llm",
                alignment_pct=accuracy,
                trust_variance=min(1.0, variance / 2.0),
                consensus_stability=success_rate,
                response_accuracy=accuracy,
                win_rate=0.0,  # Not applicable
                latency_ms=avg_latency
            )
        except Exception as e:
            logger.error(f"Error analyzing LLM subsystem: {e}")
            return self._default_metrics("llm")
    
    async def _analyze_trust_subsystem(self, cutoff_time: datetime) -> SystemMetrics:
        """Analyze trust/consensus system metrics"""
        try:
            # Get alignment data
            alignment_data = await self.db.llm_alignment.find(
                {"timestamp": {"$gte": cutoff_time.isoformat()}}
            ).to_list(1000)
            
            if alignment_data:
                alignments = [a.get("alignment_score", 0) * 100 for a in alignment_data]
                avg_alignment = np.mean(alignments)
                variance = np.std(alignments) / 100.0
            else:
                avg_alignment = 85.0
                variance = 0.05
            
            return SystemMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                subsystem="trust",
                alignment_pct=avg_alignment,
                trust_variance=variance,
                consensus_stability=1.0 - variance,
                response_accuracy=avg_alignment,
                win_rate=0.0,
                latency_ms=0
            )
        except Exception as e:
            logger.error(f"Error analyzing trust subsystem: {e}")
            return self._default_metrics("trust")
    
    async def _analyze_memory_subsystem(self, cutoff_time: datetime) -> SystemMetrics:
        """Analyze memory/distillation system metrics"""
        try:
            # Get distilled knowledge
            distilled = await self.db.llm_distilled_knowledge.find(
                {"timestamp": {"$gte": cutoff_time.isoformat()}}
            ).to_list(1000)
            
            if distilled:
                avg_confidence = np.mean([d.get("confidence_score", 0) * 100 for d in distilled])
                count = len(distilled)
            else:
                avg_confidence = 75.0
                count = 0
            
            return SystemMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                subsystem="memory",
                alignment_pct=avg_confidence,
                trust_variance=0.1,
                consensus_stability=min(1.0, count / 10.0),
                response_accuracy=avg_confidence,
                win_rate=0.0,
                latency_ms=0
            )
        except Exception as e:
            logger.error(f"Error analyzing memory subsystem: {e}")
            return self._default_metrics("memory")
    
    async def _analyze_synthesis_subsystem(self, cutoff_time: datetime) -> SystemMetrics:
        """Analyze synthesis system metrics"""
        try:
            # Get synthesis results
            synthesis = await self.db.llm_synthesis_results.find(
                {"timestamp": {"$gte": cutoff_time.isoformat()}}
            ).to_list(1000)
            
            if synthesis:
                avg_confidence = np.mean([s.get("overall_confidence", 0) * 100 for s in synthesis])
                consensus = np.mean([s.get("consensus_score", 0) for s in synthesis])
            else:
                avg_confidence = 80.0
                consensus = 0.85
            
            return SystemMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                subsystem="synthesis",
                alignment_pct=avg_confidence,
                trust_variance=0.1,
                consensus_stability=consensus,
                response_accuracy=avg_confidence,
                win_rate=0.0,
                latency_ms=0
            )
        except Exception as e:
            logger.error(f"Error analyzing synthesis subsystem: {e}")
            return self._default_metrics("synthesis")
    
    async def _analyze_training_subsystem(self, cutoff_time: datetime) -> SystemMetrics:
        """Analyze training/evaluation system metrics"""
        try:
            # Get recent evaluations
            evaluations = await self.db.model_evaluations.find(
                {"timestamp": {"$gte": cutoff_time}}
            ).to_list(100)
            
            if evaluations:
                win_rates = [e.get("challenger_win_rate", 0) for e in evaluations]
                avg_win_rate = np.mean(win_rates)
                variance = np.std(win_rates)
            else:
                avg_win_rate = 0.55
                variance = 0.05
            
            return SystemMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                subsystem="training",
                alignment_pct=avg_win_rate * 100,
                trust_variance=variance,
                consensus_stability=1.0 - variance,
                response_accuracy=avg_win_rate * 100,
                win_rate=avg_win_rate,
                latency_ms=0
            )
        except Exception as e:
            logger.error(f"Error analyzing training subsystem: {e}")
            return self._default_metrics("training")
    
    def _default_metrics(self, subsystem: str) -> SystemMetrics:
        """Return default metrics when data is unavailable"""
        return SystemMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            subsystem=subsystem,
            alignment_pct=75.0,
            trust_variance=0.1,
            consensus_stability=0.85,
            response_accuracy=75.0,
            win_rate=0.55,
            latency_ms=5000.0
        )
    
    async def derive_meta_adjustments(
        self, 
        metrics: Dict[str, SystemMetrics],
        target_alignment: float = 85.0,
        target_variance: float = 0.1
    ) -> List[MetaAdjustment]:
        """
        Derive optimal parameter adjustments based on system metrics.
        Uses ±5% adjustment window for non-critical parameters.
        
        Args:
            metrics: Current system metrics per subsystem
            target_alignment: Target alignment percentage
            target_variance: Target variance threshold
        
        Returns:
            List of proposed adjustments
        """
        adjustments = []
        
        try:
            # Get current configuration
            config_doc = await self.db.llm_config.find_one({"type": "global"})
            current_config = config_doc if config_doc else {}
            
            # Analyze each subsystem and derive adjustments
            for subsystem, metric in metrics.items():
                # Check alignment
                if metric.alignment_pct < target_alignment:
                    adj = self._adjust_for_low_alignment(subsystem, metric, current_config)
                    if adj:
                        adjustments.append(adj)
                
                # Check variance
                if metric.trust_variance > target_variance:
                    adj = self._adjust_for_high_variance(subsystem, metric, current_config)
                    if adj:
                        adjustments.append(adj)
                
                # Check latency (LLM only)
                if subsystem == "llm" and metric.latency_ms > 8000:
                    adj = self._adjust_for_high_latency(metric, current_config)
                    if adj:
                        adjustments.append(adj)
            
            logger.info(f"Derived {len(adjustments)} meta-adjustments")
            return adjustments
            
        except Exception as e:
            logger.error(f"Error deriving meta-adjustments: {e}")
            return []
    
    def _adjust_for_low_alignment(
        self, 
        subsystem: str, 
        metric: SystemMetrics, 
        config: Dict
    ) -> Optional[MetaAdjustment]:
        """Generate adjustment for low alignment"""
        try:
            if subsystem == "llm":
                # Increase prompt depth for better accuracy
                current = config.get("prompt_depth", 5)
                proposed = min(10, current + 1)
                
                return MetaAdjustment(
                    adjustment_id=str(uuid.uuid4()),
                    parameter_name="prompt_depth",
                    current_value=float(current),
                    proposed_value=float(proposed),
                    change_percent=((proposed - current) / current * 100) if current > 0 else 0,
                    reason=f"Low LLM alignment ({metric.alignment_pct:.1f}% < target)",
                    confidence=0.8,
                    is_critical=False
                )
            elif subsystem == "trust":
                # Adjust trust threshold
                current = 0.75  # Default
                proposed = min(0.85, current + 0.05)
                
                return MetaAdjustment(
                    adjustment_id=str(uuid.uuid4()),
                    parameter_name="trust_threshold",
                    current_value=current,
                    proposed_value=proposed,
                    change_percent=((proposed - current) / current * 100),
                    reason=f"Low trust alignment ({metric.alignment_pct:.1f}% < target)",
                    confidence=0.75,
                    is_critical=False
                )
        except Exception as e:
            logger.error(f"Error adjusting for low alignment: {e}")
        
        return None
    
    def _adjust_for_high_variance(
        self, 
        subsystem: str, 
        metric: SystemMetrics, 
        config: Dict
    ) -> Optional[MetaAdjustment]:
        """Generate adjustment for high variance"""
        try:
            if subsystem == "synthesis":
                # Adjust synthesis weights for stability
                current = 0.5
                proposed = min(0.6, current + 0.05)
                
                return MetaAdjustment(
                    adjustment_id=str(uuid.uuid4()),
                    parameter_name="synthesis_weight_llm",
                    current_value=current,
                    proposed_value=proposed,
                    change_percent=((proposed - current) / current * 100),
                    reason=f"High variance in synthesis ({metric.trust_variance:.2f} > target)",
                    confidence=0.7,
                    is_critical=False
                )
        except Exception as e:
            logger.error(f"Error adjusting for high variance: {e}")
        
        return None
    
    def _adjust_for_high_latency(
        self, 
        metric: SystemMetrics, 
        config: Dict
    ) -> Optional[MetaAdjustment]:
        """Generate adjustment for high latency"""
        try:
            current = config.get("max_response_time", 10.0)
            proposed = max(5.0, current - 1.0)
            
            return MetaAdjustment(
                adjustment_id=str(uuid.uuid4()),
                parameter_name="max_response_time",
                current_value=current,
                proposed_value=proposed,
                change_percent=((proposed - current) / current * 100),
                reason=f"High LLM latency ({metric.latency_ms:.0f}ms > 8000ms)",
                confidence=0.85,
                is_critical=False
            )
        except Exception as e:
            logger.error(f"Error adjusting for high latency: {e}")
        
        return None
    
    async def simulate_meta_feedback_cycle(
        self, 
        adjustments: List[MetaAdjustment]
    ) -> Dict[str, Any]:
        """
        Run sandbox simulation of proposed adjustments before live application.
        
        Returns:
            Simulation results with predicted impact
        """
        try:
            simulation_results = {
                "simulated": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "adjustments_count": len(adjustments),
                "predicted_impact": {},
                "risks": [],
                "recommendation": "approve"
            }
            
            # Simulate impact of each adjustment
            for adj in adjustments:
                impact = self._simulate_adjustment_impact(adj)
                simulation_results["predicted_impact"][adj.parameter_name] = impact
                
                # Check for risks
                if abs(adj.change_percent) > 20:
                    simulation_results["risks"].append(
                        f"Large change in {adj.parameter_name}: {adj.change_percent:.1f}%"
                    )
                
                if adj.is_critical:
                    simulation_results["recommendation"] = "manual_review"
            
            # Overall recommendation
            if len(simulation_results["risks"]) > 3:
                simulation_results["recommendation"] = "manual_review"
            
            logger.info(f"Simulation complete: {simulation_results['recommendation']}")
            return simulation_results
            
        except Exception as e:
            logger.error(f"Error simulating meta feedback cycle: {e}")
            return {
                "simulated": False,
                "error": str(e),
                "recommendation": "abort"
            }
    
    def _simulate_adjustment_impact(self, adjustment: MetaAdjustment) -> Dict[str, Any]:
        """Simulate impact of a single adjustment"""
        # Simple heuristic-based simulation
        impact = {
            "parameter": adjustment.parameter_name,
            "expected_change": adjustment.change_percent,
            "confidence": adjustment.confidence,
            "estimated_improvement": {}
        }
        
        # Heuristic predictions based on parameter type
        if "prompt_depth" in adjustment.parameter_name:
            impact["estimated_improvement"]["accuracy"] = abs(adjustment.change_percent) * 0.5
            impact["estimated_improvement"]["latency"] = -abs(adjustment.change_percent) * 0.3
        elif "threshold" in adjustment.parameter_name:
            impact["estimated_improvement"]["alignment"] = abs(adjustment.change_percent) * 0.4
            impact["estimated_improvement"]["variance"] = -abs(adjustment.change_percent) * 0.2
        elif "response_time" in adjustment.parameter_name:
            impact["estimated_improvement"]["latency"] = adjustment.change_percent
        
        return impact
    
    async def apply_autonomous_optimization(
        self, 
        adjustments: List[MetaAdjustment],
        simulation_results: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Apply approved adjustments to live system.
        Only applies non-critical parameters automatically.
        
        Returns:
            Tuple of (success, application_results)
        """
        try:
            # Check simulation recommendation
            if simulation_results.get("recommendation") == "abort":
                return False, {"error": "Simulation recommended abort"}
            
            if simulation_results.get("recommendation") == "manual_review":
                logger.info("Manual review required - storing for approval")
                return False, {
                    "status": "pending_approval",
                    "message": "Manual review required for critical changes"
                }
            
            # Apply adjustments
            applied_count = 0
            failed_count = 0
            results = {
                "applied": [],
                "failed": [],
                "skipped": []
            }
            
            for adj in adjustments:
                if adj.is_critical:
                    results["skipped"].append({
                        "parameter": adj.parameter_name,
                        "reason": "Critical parameter requires manual approval"
                    })
                    continue
                
                try:
                    success = await self._apply_single_adjustment(adj)
                    if success:
                        applied_count += 1
                        results["applied"].append(adj.parameter_name)
                    else:
                        failed_count += 1
                        results["failed"].append(adj.parameter_name)
                except Exception as e:
                    logger.error(f"Error applying adjustment {adj.parameter_name}: {e}")
                    failed_count += 1
                    results["failed"].append(adj.parameter_name)
            
            logger.info(f"Applied {applied_count} adjustments, {failed_count} failed, {len(results['skipped'])} skipped")
            
            return True, {
                "success": True,
                "applied_count": applied_count,
                "failed_count": failed_count,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error applying autonomous optimization: {e}")
            return False, {"error": str(e)}
    
    async def _apply_single_adjustment(self, adjustment: MetaAdjustment) -> bool:
        """Apply a single parameter adjustment"""
        try:
            # Update configuration in database
            if adjustment.parameter_name in ["prompt_depth", "max_response_time"]:
                # LLM config
                await self.db.llm_config.update_one(
                    {"type": "global"},
                    {"$set": {
                        adjustment.parameter_name: adjustment.proposed_value,
                        "updated_at": datetime.now(timezone.utc)
                    }},
                    upsert=True
                )
            else:
                # Store in meta_config collection
                await self.db.meta_config.update_one(
                    {"parameter": adjustment.parameter_name},
                    {"$set": {
                        "value": adjustment.proposed_value,
                        "updated_at": datetime.now(timezone.utc),
                        "reason": adjustment.reason
                    }},
                    upsert=True
                )
            
            logger.info(f"Applied adjustment: {adjustment.parameter_name} = {adjustment.proposed_value}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying adjustment: {e}")
            return False
    
    async def run_optimization_cycle(
        self, 
        trigger: str = "scheduled"
    ) -> OptimizationCycle:
        """
        Run complete meta-optimization cycle.
        
        Args:
            trigger: "scheduled", "manual", or "performance_threshold"
        
        Returns:
            OptimizationCycle with complete results
        """
        cycle_id = str(uuid.uuid4())
        logger.info(f"Starting optimization cycle {cycle_id} (trigger: {trigger})")
        
        try:
            # Step 1: Analyze system performance
            metrics = await self.analyze_system_performance(lookback_hours=24)
            
            # Step 2: Derive meta-adjustments
            adjustments = await self.derive_meta_adjustments(metrics)
            
            # Step 3: Simulate adjustments
            simulation_results = await self.simulate_meta_feedback_cycle(adjustments)
            
            # Step 4: Apply if approved
            applied = False
            approval_required = False
            
            if simulation_results.get("recommendation") == "approve":
                success, application_results = await self.apply_autonomous_optimization(
                    adjustments, 
                    simulation_results
                )
                applied = success
            elif simulation_results.get("recommendation") == "manual_review":
                approval_required = True
            
            # Step 5: Calculate performance delta
            performance_delta = self._calculate_performance_delta(metrics, adjustments)
            
            # Step 6: Generate reflection summary
            reflection = self._generate_reflection_summary(
                metrics, adjustments, simulation_results, applied
            )
            
            # Step 7: Calculate system health score
            health_score = self._calculate_system_health(metrics)
            
            # Create optimization cycle result
            cycle = OptimizationCycle(
                cycle_id=cycle_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                trigger=trigger,
                adjustments=adjustments,
                simulation_results=simulation_results,
                applied=applied,
                approval_required=approval_required,
                performance_delta=performance_delta,
                reflection_summary=reflection,
                system_health_score=health_score
            )
            
            # Store in database
            await self.db.llm_meta_optimization_log.insert_one(cycle.to_dict())
            
            self.last_optimization = datetime.now(timezone.utc)
            logger.info(f"Optimization cycle {cycle_id} complete: applied={applied}, health={health_score:.1f}")
            
            return cycle
            
        except Exception as e:
            logger.error(f"Error in optimization cycle: {e}")
            return OptimizationCycle(
                cycle_id=cycle_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                trigger=trigger,
                adjustments=[],
                simulation_results={"error": str(e)},
                applied=False,
                approval_required=False,
                performance_delta={},
                reflection_summary=f"Cycle failed: {str(e)}",
                system_health_score=50.0
            )
    
    def _calculate_performance_delta(
        self, 
        metrics: Dict[str, SystemMetrics],
        adjustments: List[MetaAdjustment]
    ) -> Dict[str, float]:
        """Calculate expected performance changes"""
        delta = {}
        
        for adj in adjustments:
            if "prompt_depth" in adj.parameter_name:
                delta["accuracy"] = adj.change_percent * 0.5
                delta["latency"] = -adj.change_percent * 0.3
            elif "threshold" in adj.parameter_name:
                delta["alignment"] = adj.change_percent * 0.4
                delta["variance"] = -adj.change_percent * 0.2
        
        return delta
    
    def _generate_reflection_summary(
        self,
        metrics: Dict[str, SystemMetrics],
        adjustments: List[MetaAdjustment],
        simulation_results: Dict[str, Any],
        applied: bool
    ) -> str:
        """Generate human-readable reflection summary"""
        
        avg_alignment = np.mean([m.alignment_pct for m in metrics.values()])
        avg_variance = np.mean([m.trust_variance for m in metrics.values()])
        
        summary_parts = []
        summary_parts.append(f"Cycle completed.")
        summary_parts.append(f"System alignment: {avg_alignment:.1f}%")
        summary_parts.append(f"Trust variance: {avg_variance:.3f}")
        
        if adjustments:
            summary_parts.append(f"Proposed {len(adjustments)} adjustments")
            if applied:
                summary_parts.append("Adjustments applied successfully")
            else:
                summary_parts.append("Adjustments pending approval")
        else:
            summary_parts.append("No adjustments needed")
        
        summary_parts.append(f"Overall stability: {'High' if avg_variance < 0.1 else 'Moderate'}")
        
        return ". ".join(summary_parts) + "."
    
    def _calculate_system_health(self, metrics: Dict[str, SystemMetrics]) -> float:
        """Calculate overall system health score (0-100)"""
        try:
            scores = []
            
            for metric in metrics.values():
                # Component scores
                alignment_score = metric.alignment_pct
                stability_score = (1.0 - metric.trust_variance) * 100
                consensus_score = metric.consensus_stability * 100
                
                # Weighted average for this subsystem
                subsystem_score = (
                    alignment_score * 0.4 +
                    stability_score * 0.3 +
                    consensus_score * 0.3
                )
                scores.append(subsystem_score)
            
            # Overall health is average of subsystem scores
            health = np.mean(scores) if scores else 75.0
            return round(min(100.0, max(0.0, health)), 2)
            
        except Exception as e:
            logger.error(f"Error calculating system health: {e}")
            return 75.0
    
    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization system status"""
        try:
            # Get latest metrics
            metrics = await self.analyze_system_performance(lookback_hours=1)
            
            # Get recent optimization cycles
            recent_cycles = await self.db.llm_meta_optimization_log.find().sort(
                "timestamp", -1
            ).limit(5).to_list(5)
            
            # Calculate status
            health = self._calculate_system_health(metrics)
            
            time_since_last = None
            if self.last_optimization:
                delta = datetime.now(timezone.utc) - self.last_optimization
                time_since_last = delta.total_seconds()
            
            return {
                "system_health_score": health,
                "health_status": self._get_health_status(health),
                "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None,
                "time_since_last_seconds": time_since_last,
                "recent_cycles_count": len(recent_cycles),
                "running": self.running,
                "optimization_interval": self.optimization_interval,
                "metrics_summary": {
                    subsystem: {
                        "alignment": metric.alignment_pct,
                        "variance": metric.trust_variance,
                        "stability": metric.consensus_stability
                    }
                    for subsystem, metric in metrics.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting optimization status: {e}")
            return {
                "system_health_score": 75.0,
                "health_status": "unknown",
                "error": str(e)
            }
    
    def _get_health_status(self, health_score: float) -> str:
        """Convert health score to status string"""
        if health_score >= 85:
            return "excellent"
        elif health_score >= 70:
            return "good"
        elif health_score >= 50:
            return "moderate"
        else:
            return "critical"
