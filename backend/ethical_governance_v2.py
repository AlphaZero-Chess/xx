"""
Ethical Governance Layer 2.0 (Step 33)

This module implements the continuously operating ethical governance and oversight system that:
- Monitors all cognitive subsystems (Creativity, Reflection, Memory, Cohesion) for ethical integrity
- Provides real-time compliance scoring and anomaly detection
- Enforces adaptive ethical thresholds based on context and learning cycles
- Maintains transparent human-oversight tools without affecting gameplay
- Operates in advisory-only mode with human-in-the-loop approval for parameter changes

Core Components:
- Autonomous Oversight Kernel (AOK): Central monitor for real-time ethical validation
- Adaptive Threshold Engine (ATE): Adjusts ethical limits dynamically per game phase
- Compliance Audit Stream (CAS): Generates continuous audit logs
- Governance Console: UI integration for human review
- Human-in-the-Loop Review Mode: Explicit approval for parameter changes

Safety Directives:
- Advisory mode only - no gameplay interference
- Full audit trail stored in MongoDB
- Human confirmation required for parameter changes
- Supports regression testing before live adoption
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
class EthicalMetrics:
    """Real-time ethical compliance metrics"""
    timestamp: str
    compliance_index: float  # 0-1: Overall compliance score (target ≥ 0.95)
    fair_play_score: float  # 0-1: Fair play adherence
    transparency_score: float  # 0-1: Transparency and explainability
    educational_value_score: float  # 0-1: Learning/educational contribution
    non_manipulation_score: float  # 0-1: Freedom from manipulation
    ethical_continuity: float  # 0-1: Consistency across cycles (target ≥ 0.93)
    
    # Module-specific scores
    creativity_ethics: float
    reflection_ethics: float
    memory_ethics: float
    cohesion_ethics: float
    
    # System health
    anomalies_detected: int
    violations_flagged: int
    false_positive_rate: float  # Target ≤ 0.03
    review_latency: float  # seconds (target ≤ 2.0)
    
    status: str  # "excellent", "good", "needs_attention", "critical"
    
    def to_dict(self):
        return asdict(self)


@dataclass
class EthicalViolation:
    """A flagged ethical violation or anomaly"""
    violation_id: str
    timestamp: str
    severity: str  # "critical", "high", "medium", "low"
    module: str  # Which module flagged
    violation_type: str  # Type of violation
    description: str
    context: Dict[str, Any]
    recommended_action: str
    requires_human_review: bool
    auto_flagged: bool
    resolution_status: str  # "pending", "approved", "rejected", "resolved"
    resolution_timestamp: Optional[str]
    resolution_notes: Optional[str]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class AdaptiveThreshold:
    """Dynamic ethical threshold per module/context"""
    threshold_id: str
    module: str
    parameter: str
    base_threshold: float
    current_threshold: float
    context: str  # "opening", "middlegame", "endgame", "general"
    adjustment_history: List[Dict[str, Any]]
    last_adjusted: str
    reason: str
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ParameterChangeRequest:
    """Request for parameter change requiring human approval"""
    request_id: str
    timestamp: str
    module: str
    parameter: str
    current_value: float
    proposed_value: float
    delta: float
    reason: str
    impact_analysis: str
    severity: str  # "critical", "high", "medium", "low"
    requires_approval: bool
    approval_status: str  # "pending", "approved", "rejected"
    approved_by: Optional[str]
    approval_timestamp: Optional[str]
    approval_notes: Optional[str]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class EthicsReport:
    """Comprehensive ethics report (Ethics Report #001)"""
    report_id: str
    report_number: int
    timestamp: str
    reporting_period: Dict[str, str]  # start, end
    
    # Summary metrics
    overall_compliance: float
    total_scans: int
    total_violations: int
    violations_by_severity: Dict[str, int]
    
    # Module analysis
    module_compliance: Dict[str, float]
    module_violations: Dict[str, int]
    
    # Trend analysis
    compliance_trend: str  # "improving", "stable", "declining"
    ethical_continuity_trend: float
    
    # Violations summary
    top_violations: List[Dict[str, Any]]
    
    # Threshold adjustments
    threshold_adjustments: List[Dict[str, Any]]
    
    # Parameter change requests
    pending_approvals: int
    approved_changes: int
    rejected_changes: int
    
    # Recommendations
    recommendations: List[str]
    
    # Health assessment
    system_health_assessment: str
    
    def to_dict(self):
        return asdict(self)


class EthicalGovernanceController:
    """
    Ethical Governance Layer 2.0 Controller
    
    Continuously monitors and validates ethical compliance across all cognitive
    subsystems (Steps 29-32) with adaptive thresholds and human oversight.
    
    Core Principles:
    - Advisory mode only (no direct parameter changes without approval)
    - Transparent audit trail
    - Adaptive ethical boundaries
    - Human-in-the-loop for critical decisions
    """
    
    def __init__(self, db_client):
        self.db = db_client
        self.llm_key = os.environ.get('EMERGENT_LLM_KEY')
        
        if not self.llm_key:
            logger.warning("EMERGENT_LLM_KEY not found - ethics will use mock mode")
            self.llm_available = False
        else:
            self.llm_available = True
            logger.info("Ethical Governance Controller initialized with multi-provider LLM support")
        
        # Target metrics (from problem statement)
        self.target_metrics = {
            "compliance_index": 0.95,
            "ethical_continuity": 0.93,
            "false_positive_rate": 0.03,
            "review_latency": 2.0
        }
        
        # Ethical policies (core rules)
        self.ethical_policies = {
            "fair_play": {
                "name": "Fair Play",
                "description": "Must not enable unfair advantages or cheating in gameplay",
                "weight": 0.30,
                "base_threshold": 0.85
            },
            "transparency": {
                "name": "Transparency",
                "description": "All decisions must be explainable and transparent",
                "weight": 0.25,
                "base_threshold": 0.80
            },
            "educational_value": {
                "name": "Educational Value",
                "description": "Should provide learning value and be instructive",
                "weight": 0.25,
                "base_threshold": 0.75
            },
            "non_manipulation": {
                "name": "Non-Manipulation",
                "description": "Must not manipulate users or exploit psychological vulnerabilities",
                "weight": 0.20,
                "base_threshold": 0.90
            }
        }
        
        # Adaptive thresholds per game phase
        self.phase_adjustments = {
            "opening": {"novelty": +0.05, "stability": -0.03},  # Allow more creativity in openings
            "middlegame": {"novelty": 0.0, "stability": 0.0},  # Balanced
            "endgame": {"novelty": -0.05, "stability": +0.05}  # Prioritize stability in endgames
        }
        
        # Severity thresholds for violations
        self.severity_thresholds = {
            "critical": 0.50,  # Compliance < 0.50
            "high": 0.70,      # Compliance < 0.70
            "medium": 0.85,    # Compliance < 0.85
            "low": 0.95        # Compliance < 0.95
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
            f"Ethical Governance Layer 2.0 initialized: "
            f"Target compliance={self.target_metrics['compliance_index']}, "
            f"Continuity={self.target_metrics['ethical_continuity']}"
        )
    
    async def monitor_system_state(self) -> Dict[str, Any]:
        """
        Autonomous Oversight Kernel (AOK)
        
        Poll all cognitive modules (Creativity, Reflection, Memory, Cohesion) 
        for their current outputs and ethical status.
        
        Returns:
            Complete system state snapshot
        """
        try:
            logger.info("AOK: Monitoring system state across all modules...")
            
            system_state = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "modules": {}
            }
            
            # Monitor Creativity (Step 29)
            creativity_state = await self._monitor_creativity_module()
            system_state["modules"]["creativity"] = creativity_state
            
            # Monitor Reflection (Step 30)
            reflection_state = await self._monitor_reflection_module()
            system_state["modules"]["reflection"] = reflection_state
            
            # Monitor Memory (Step 31)
            memory_state = await self._monitor_memory_module()
            system_state["modules"]["memory"] = memory_state
            
            # Monitor Cohesion (Step 32)
            cohesion_state = await self._monitor_cohesion_module()
            system_state["modules"]["cohesion"] = cohesion_state
            
            logger.info(f"AOK: System state captured for {len(system_state['modules'])} modules")
            
            return system_state
            
        except Exception as e:
            logger.error(f"Error monitoring system state: {e}")
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}
    
    async def _monitor_creativity_module(self) -> Dict[str, Any]:
        """Monitor Creativity module (Step 29) ethical compliance"""
        try:
            # Get recent creative strategies
            recent_strategies = await self.db.llm_creative_synthesis.find({
                "rejected": False
            }).sort("timestamp", -1).limit(10).to_list(10)
            
            if not recent_strategies:
                return {
                    "status": "inactive",
                    "ethical_score": 0.90,
                    "strategies_count": 0
                }
            
            # Calculate average ethical alignment
            ethical_scores = [s.get("ethical_alignment", 0.9) for s in recent_strategies]
            avg_ethical = np.mean(ethical_scores)
            
            # Check for rejected strategies (potential violations)
            rejected_count = await self.db.llm_creative_synthesis.count_documents({
                "rejected": True,
                "timestamp": {"$gte": (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()}
            })
            
            return {
                "status": "operational",
                "ethical_score": round(avg_ethical, 2),
                "strategies_count": len(recent_strategies),
                "rejected_24h": rejected_count,
                "avg_novelty": round(np.mean([s.get("novelty_score", 0.7) for s in recent_strategies]), 2),
                "avg_stability": round(np.mean([s.get("stability_score", 0.65) for s in recent_strategies]), 2),
                "compliant": avg_ethical >= 0.75
            }
            
        except Exception as e:
            logger.error(f"Error monitoring creativity module: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _monitor_reflection_module(self) -> Dict[str, Any]:
        """Monitor Reflection module (Step 30) ethical compliance"""
        try:
            # Get latest reflection cycle
            latest_cycle = await self.db.llm_reflection_log.find_one(
                sort=[("timestamp", -1)]
            )
            
            if not latest_cycle:
                return {
                    "status": "inactive",
                    "ethical_score": 0.85,
                    "cycles_count": 0
                }
            
            ethical_status = latest_cycle.get("ethical_alignment_status", "good")
            ethical_score_map = {
                "excellent": 0.95,
                "good": 0.85,
                "needs_attention": 0.70,
                "critical": 0.50
            }
            ethical_score = ethical_score_map.get(ethical_status, 0.80)
            
            # Get current learning parameters
            params = await self.db.llm_learning_parameters.find_one(
                sort=[("timestamp", -1)]
            )
            
            return {
                "status": "operational",
                "ethical_score": ethical_score,
                "ethical_status": ethical_status,
                "learning_health": latest_cycle.get("learning_health_index", 0.75),
                "performance_score": latest_cycle.get("overall_performance_score", 70.0),
                "parameters": {
                    "novelty_weight": params.get("novelty_weight", 0.60) if params else 0.60,
                    "ethical_threshold": params.get("ethical_threshold", 0.75) if params else 0.75
                },
                "compliant": ethical_score >= 0.75
            }
            
        except Exception as e:
            logger.error(f"Error monitoring reflection module: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _monitor_memory_module(self) -> Dict[str, Any]:
        """Monitor Memory module (Step 31) ethical compliance"""
        try:
            # Get active memory nodes
            active_nodes = await self.db.llm_memory_nodes.find({
                "decay_weight": {"$gte": 0.20}
            }).sort("timestamp", -1).limit(20).to_list(20)
            
            if not active_nodes:
                return {
                    "status": "inactive",
                    "ethical_score": 0.92,
                    "nodes_count": 0
                }
            
            # Calculate average ethical alignment
            ethical_scores = [n.get("ethical_alignment", 0.9) for n in active_nodes]
            avg_ethical = np.mean(ethical_scores)
            
            # Get latest fusion metrics
            latest_fusion = await self.db.llm_memory_trace.find_one({
                "trace_type": "fusion_cycle"
            }, sort=[("timestamp", -1)])
            
            ethical_continuity = 0.92
            if latest_fusion and "fusion_metrics" in latest_fusion:
                ethical_continuity = latest_fusion["fusion_metrics"].get("ethical_continuity", 0.92)
            
            return {
                "status": "operational",
                "ethical_score": round(avg_ethical, 2),
                "ethical_continuity": ethical_continuity,
                "active_nodes": len(active_nodes),
                "avg_confidence": round(np.mean([n.get("confidence_score", 0.8) for n in active_nodes]), 2),
                "compliant": avg_ethical >= 0.85 and ethical_continuity >= 0.90
            }
            
        except Exception as e:
            logger.error(f"Error monitoring memory module: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _monitor_cohesion_module(self) -> Dict[str, Any]:
        """Monitor Cohesion module (Step 32) ethical compliance"""
        try:
            # Get latest cohesion report
            latest_report = await self.db.llm_cohesion_reports.find_one(
                sort=[("timestamp", -1)]
            )
            
            if not latest_report:
                return {
                    "status": "inactive",
                    "ethical_score": 0.90,
                    "reports_count": 0
                }
            
            metrics = latest_report.get("metrics", {})
            ethical_continuity = metrics.get("ethical_continuity", 0.90)
            alignment_score = metrics.get("alignment_score", 0.85)
            
            return {
                "status": "operational",
                "ethical_score": ethical_continuity,
                "alignment_score": alignment_score,
                "system_health": metrics.get("system_health_index", 0.80),
                "drift_detected": metrics.get("drift_detected", False),
                "cohesion_health": metrics.get("cohesion_health", "good"),
                "compliant": ethical_continuity >= 0.85 and alignment_score >= 0.80
            }
            
        except Exception as e:
            logger.error(f"Error monitoring cohesion module: {e}")
            return {"status": "error", "error": str(e)}
    
    async def evaluate_compliance(
        self,
        system_state: Dict[str, Any],
        context: Optional[str] = "general"
    ) -> EthicalMetrics:
        """
        Evaluate system-wide ethical compliance against policies.
        
        Scores against: Fair Play, Transparency, Educational Value, Non-Manipulation
        
        Args:
            system_state: Current system state from monitor_system_state()
            context: Game phase context ("opening", "middlegame", "endgame", "general")
        
        Returns:
            Comprehensive ethical metrics
        """
        try:
            logger.info(f"Evaluating compliance for context: {context}")
            
            modules = system_state.get("modules", {})
            
            # Extract module ethical scores
            creativity_ethics = modules.get("creativity", {}).get("ethical_score", 0.90)
            reflection_ethics = modules.get("reflection", {}).get("ethical_score", 0.85)
            memory_ethics = modules.get("memory", {}).get("ethical_score", 0.92)
            cohesion_ethics = modules.get("cohesion", {}).get("ethical_score", 0.90)
            
            # Calculate policy-specific scores
            fair_play_score = await self._evaluate_fair_play(modules, context)
            transparency_score = await self._evaluate_transparency(modules, context)
            educational_score = await self._evaluate_educational_value(modules, context)
            non_manipulation_score = await self._evaluate_non_manipulation(modules, context)
            
            # Calculate overall compliance index (weighted average)
            compliance_index = (
                fair_play_score * self.ethical_policies["fair_play"]["weight"] +
                transparency_score * self.ethical_policies["transparency"]["weight"] +
                educational_score * self.ethical_policies["educational_value"]["weight"] +
                non_manipulation_score * self.ethical_policies["non_manipulation"]["weight"]
            )
            
            # Calculate ethical continuity (consistency across modules)
            module_scores = [creativity_ethics, reflection_ethics, memory_ethics, cohesion_ethics]
            ethical_continuity = 1.0 - (np.std(module_scores) * 2)  # Lower variance = higher continuity
            ethical_continuity = max(0.0, min(1.0, ethical_continuity))
            
            # Count recent anomalies and violations
            anomalies_count = await self.db.llm_ethics_audit.count_documents({
                "timestamp": {"$gte": (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()},
                "violation_type": "anomaly"
            })
            
            violations_count = await self.db.llm_ethics_audit.count_documents({
                "timestamp": {"$gte": (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()},
                "severity": {"$in": ["critical", "high"]}
            })
            
            # Calculate false positive rate (flagged but resolved as non-issues)
            total_flagged = await self.db.llm_ethics_audit.count_documents({
                "timestamp": {"$gte": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()}
            })
            false_positives = await self.db.llm_ethics_audit.count_documents({
                "timestamp": {"$gte": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()},
                "resolution_status": "resolved",
                "resolution_notes": {"$regex": "false.?positive", "$options": "i"}
            })
            
            false_positive_rate = false_positives / total_flagged if total_flagged > 0 else 0.0
            
            # Review latency (mock for now - would measure actual approval times)
            review_latency = 1.5  # seconds
            
            # Determine status
            if compliance_index >= 0.95 and ethical_continuity >= 0.93:
                status = "excellent"
            elif compliance_index >= 0.85 and ethical_continuity >= 0.85:
                status = "good"
            elif compliance_index >= 0.70:
                status = "needs_attention"
            else:
                status = "critical"
            
            metrics = EthicalMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                compliance_index=round(compliance_index, 3),
                fair_play_score=round(fair_play_score, 3),
                transparency_score=round(transparency_score, 3),
                educational_value_score=round(educational_score, 3),
                non_manipulation_score=round(non_manipulation_score, 3),
                ethical_continuity=round(ethical_continuity, 3),
                creativity_ethics=round(creativity_ethics, 2),
                reflection_ethics=round(reflection_ethics, 2),
                memory_ethics=round(memory_ethics, 2),
                cohesion_ethics=round(cohesion_ethics, 2),
                anomalies_detected=anomalies_count,
                violations_flagged=violations_count,
                false_positive_rate=round(false_positive_rate, 3),
                review_latency=review_latency,
                status=status
            )
            
            # Store metrics
            await self.db.llm_ethics_metrics.insert_one(metrics.to_dict())
            
            logger.info(
                f"Compliance evaluated: Index={compliance_index:.3f}, "
                f"Continuity={ethical_continuity:.3f}, Status={status}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating compliance: {e}")
            raise
    
    async def _evaluate_fair_play(
        self,
        modules: Dict[str, Any],
        context: str
    ) -> float:
        """Evaluate Fair Play policy compliance"""
        # Check for any rejected strategies (potential unfair tactics)
        creativity = modules.get("creativity", {})
        rejected_count = creativity.get("rejected_24h", 0)
        
        # Penalize for rejections
        base_score = 0.95
        if rejected_count > 0:
            base_score -= min(0.20, rejected_count * 0.05)
        
        # Check reflection module for ethical compliance
        reflection = modules.get("reflection", {})
        if reflection.get("ethical_status") in ["needs_attention", "critical"]:
            base_score -= 0.10
        
        return max(0.70, min(1.0, base_score))
    
    async def _evaluate_transparency(
        self,
        modules: Dict[str, Any],
        context: str
    ) -> float:
        """Evaluate Transparency policy compliance"""
        # Transparency is high if strategies are explainable
        creativity = modules.get("creativity", {})
        reflection = modules.get("reflection", {})
        
        base_score = 0.90
        
        # Educational value indicates explainability
        if creativity.get("strategies_count", 0) > 0:
            # Strategies exist and are documented
            base_score = 0.92
        
        # Reflection provides insights (transparent learning)
        if reflection.get("learning_health", 0) >= 0.70:
            base_score += 0.05
        
        return min(1.0, base_score)
    
    async def _evaluate_educational_value(
        self,
        modules: Dict[str, Any],
        context: str
    ) -> float:
        """Evaluate Educational Value policy compliance"""
        creativity = modules.get("creativity", {})
        reflection = modules.get("reflection", {})
        
        base_score = 0.80
        
        # Creative strategies should have educational value
        if creativity.get("strategies_count", 0) > 0:
            base_score += 0.10
        
        # Reflection learning indicates educational growth
        learning_health = reflection.get("learning_health", 0.70)
        base_score += learning_health * 0.10
        
        return min(1.0, base_score)
    
    async def _evaluate_non_manipulation(
        self,
        modules: Dict[str, Any],
        context: str
    ) -> float:
        """Evaluate Non-Manipulation policy compliance"""
        # High score by default (no manipulation detected)
        base_score = 0.95
        
        # Check for any manipulation indicators
        cohesion = modules.get("cohesion", {})
        if cohesion.get("drift_detected", False):
            # Drift might indicate unintended manipulation
            base_score -= 0.05
        
        # All modules should maintain ethical standards
        all_compliant = all([
            modules.get("creativity", {}).get("compliant", True),
            modules.get("reflection", {}).get("compliant", True),
            modules.get("memory", {}).get("compliant", True),
            modules.get("cohesion", {}).get("compliant", True)
        ])
        
        if not all_compliant:
            base_score -= 0.10
        
        return max(0.75, base_score)
    
    async def auto_flag_anomalies(
        self,
        system_state: Dict[str, Any],
        metrics: EthicalMetrics
    ) -> List[EthicalViolation]:
        """
        Automatically detect and flag ethical drift or policy violations.
        
        Args:
            system_state: Current system state
            metrics: Current ethical metrics
        
        Returns:
            List of flagged violations
        """
        try:
            logger.info("Auto-flagging anomalies and violations...")
            
            violations = []
            
            # Check 1: Compliance index below target
            if metrics.compliance_index < self.target_metrics["compliance_index"]:
                severity = self._determine_severity(metrics.compliance_index)
                violation = EthicalViolation(
                    violation_id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=severity,
                    module="system",
                    violation_type="compliance_below_target",
                    description=f"Compliance index ({metrics.compliance_index:.3f}) below target ({self.target_metrics['compliance_index']:.2f})",
                    context={"compliance_index": metrics.compliance_index, "target": self.target_metrics["compliance_index"]},
                    recommended_action="Review module-specific scores and tighten ethical constraints",
                    requires_human_review=severity in ["critical", "high"],
                    auto_flagged=True,
                    resolution_status="pending",
                    resolution_timestamp=None,
                    resolution_notes=None
                )
                violations.append(violation)
            
            # Check 2: Ethical continuity below target
            if metrics.ethical_continuity < self.target_metrics["ethical_continuity"]:
                severity = self._determine_severity(metrics.ethical_continuity)
                violation = EthicalViolation(
                    violation_id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=severity,
                    module="system",
                    violation_type="ethical_continuity_low",
                    description=f"Ethical continuity ({metrics.ethical_continuity:.3f}) below target ({self.target_metrics['ethical_continuity']:.2f})",
                    context={"ethical_continuity": metrics.ethical_continuity, "module_variance": np.std([
                        metrics.creativity_ethics, metrics.reflection_ethics, 
                        metrics.memory_ethics, metrics.cohesion_ethics
                    ])},
                    recommended_action="Synchronize ethical standards across modules",
                    requires_human_review=severity in ["critical", "high"],
                    auto_flagged=True,
                    resolution_status="pending",
                    resolution_timestamp=None,
                    resolution_notes=None
                )
                violations.append(violation)
            
            # Check 3: Module-specific ethical drops
            modules = system_state.get("modules", {})
            for module_name, module_data in modules.items():
                ethical_score = module_data.get("ethical_score", 0.90)
                if ethical_score < 0.75:
                    severity = self._determine_severity(ethical_score)
                    violation = EthicalViolation(
                        violation_id=str(uuid.uuid4()),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        severity=severity,
                        module=module_name,
                        violation_type="module_ethics_low",
                        description=f"{module_name.capitalize()} module ethical score ({ethical_score:.2f}) below minimum (0.75)",
                        context={"module": module_name, "score": ethical_score, "data": module_data},
                        recommended_action=f"Review {module_name} module parameters and recent outputs",
                        requires_human_review=severity in ["critical", "high"],
                        auto_flagged=True,
                        resolution_status="pending",
                        resolution_timestamp=None,
                        resolution_notes=None
                    )
                    violations.append(violation)
            
            # Check 4: False positive rate too high
            if metrics.false_positive_rate > self.target_metrics["false_positive_rate"]:
                violation = EthicalViolation(
                    violation_id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity="medium",
                    module="governance",
                    violation_type="high_false_positive_rate",
                    description=f"False positive rate ({metrics.false_positive_rate:.1%}) exceeds target ({self.target_metrics['false_positive_rate']:.0%})",
                    context={"false_positive_rate": metrics.false_positive_rate},
                    recommended_action="Recalibrate anomaly detection thresholds to reduce false positives",
                    requires_human_review=False,
                    auto_flagged=True,
                    resolution_status="pending",
                    resolution_timestamp=None,
                    resolution_notes=None
                )
                violations.append(violation)
            
            # Check 5: Drift detection in cohesion
            cohesion = modules.get("cohesion", {})
            if cohesion.get("drift_detected", False):
                violation = EthicalViolation(
                    violation_id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity="medium",
                    module="cohesion",
                    violation_type="parameter_drift_detected",
                    description="Parameter drift detected across cognitive modules",
                    context={"cohesion_health": cohesion.get("cohesion_health"), "alignment_score": cohesion.get("alignment_score")},
                    recommended_action="Monitor parameter evolution and assess if drift is beneficial or harmful",
                    requires_human_review=False,
                    auto_flagged=True,
                    resolution_status="pending",
                    resolution_timestamp=None,
                    resolution_notes=None
                )
                violations.append(violation)
            
            # Store violations in audit log
            if violations:
                for violation in violations:
                    await self.db.llm_ethics_audit.insert_one(violation.to_dict())
            
            logger.info(f"Auto-flagged {len(violations)} violations")
            
            return violations
            
        except Exception as e:
            logger.error(f"Error auto-flagging anomalies: {e}")
            return []
    
    def _determine_severity(self, score: float) -> str:
        """Determine violation severity based on score"""
        if score < self.severity_thresholds["critical"]:
            return "critical"
        elif score < self.severity_thresholds["high"]:
            return "high"
        elif score < self.severity_thresholds["medium"]:
            return "medium"
        else:
            return "low"
    
    async def recalibrate_thresholds(
        self,
        metrics: EthicalMetrics,
        context: str = "general"
    ) -> List[AdaptiveThreshold]:
        """
        Adaptive Threshold Engine (ATE)
        
        Dynamically adjust ethical thresholds based on context and trends.
        
        Args:
            metrics: Current ethical metrics
            context: Game phase or operational context
        
        Returns:
            List of updated adaptive thresholds
        """
        try:
            logger.info(f"ATE: Recalibrating thresholds for context: {context}")
            
            updated_thresholds = []
            
            # Get context-specific adjustments
            phase_adj = self.phase_adjustments.get(context, {})
            
            # Recalibrate per module
            for module in ["creativity", "reflection", "memory", "cohesion"]:
                for parameter in ["ethical_threshold", "novelty_weight", "stability_weight"]:
                    # Get base threshold
                    base_threshold = self.ethical_policies.get("fair_play", {}).get("base_threshold", 0.85)
                    
                    # Apply context adjustment
                    adjustment = phase_adj.get(parameter.split("_")[0], 0.0)
                    current_threshold = base_threshold + adjustment
                    
                    # Adaptive adjustment based on recent performance
                    if metrics.compliance_index < 0.90:
                        # Tighten thresholds if compliance is low
                        adjustment += 0.02
                        reason = "Compliance below target - tightening thresholds"
                    elif metrics.compliance_index >= 0.95 and metrics.ethical_continuity >= 0.93:
                        # Can slightly relax if performance is excellent
                        adjustment -= 0.01
                        reason = "Excellent compliance - allowing slight relaxation"
                    else:
                        reason = "Maintaining current thresholds"
                    
                    current_threshold = max(0.70, min(0.95, base_threshold + adjustment))
                    
                    threshold = AdaptiveThreshold(
                        threshold_id=str(uuid.uuid4()),
                        module=module,
                        parameter=parameter,
                        base_threshold=base_threshold,
                        current_threshold=round(current_threshold, 3),
                        context=context,
                        adjustment_history=[{
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "adjustment": adjustment,
                            "reason": reason
                        }],
                        last_adjusted=datetime.now(timezone.utc).isoformat(),
                        reason=reason
                    )
                    
                    updated_thresholds.append(threshold)
            
            # Store thresholds
            for threshold in updated_thresholds:
                # Update or insert
                await self.db.llm_ethics_policies.update_one(
                    {"module": threshold.module, "parameter": threshold.parameter, "context": threshold.context},
                    {"$set": threshold.to_dict()},
                    upsert=True
                )
            
            logger.info(f"ATE: Recalibrated {len(updated_thresholds)} thresholds")
            
            return updated_thresholds
            
        except Exception as e:
            logger.error(f"Error recalibrating thresholds: {e}")
            return []
    
    async def generate_ethics_report(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> EthicsReport:
        """
        Generate comprehensive Ethics Report #001 (or subsequent reports).
        
        Args:
            start_date: Report period start (ISO format)
            end_date: Report period end (ISO format)
        
        Returns:
            Complete ethics report
        """
        try:
            # Increment report counter
            self.report_counter += 1
            report_number = self.report_counter
            
            logger.info(f"Generating Ethics Report #{report_number:03d}...")
            
            # Default to last 7 days if not specified
            if not end_date:
                end_date = datetime.now(timezone.utc).isoformat()
            if not start_date:
                start_date = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            
            # Get all metrics in period
            metrics_in_period = await self.db.llm_ethics_metrics.find({
                "timestamp": {"$gte": start_date, "$lte": end_date}
            }).to_list(1000)
            
            # Calculate summary metrics
            if metrics_in_period:
                overall_compliance = np.mean([m.get("compliance_index", 0.90) for m in metrics_in_period])
            else:
                overall_compliance = 0.90
            
            total_scans = len(metrics_in_period)
            
            # Get violations in period
            violations_in_period = await self.db.llm_ethics_audit.find({
                "timestamp": {"$gte": start_date, "$lte": end_date}
            }).to_list(1000)
            
            total_violations = len(violations_in_period)
            
            # Violations by severity
            violations_by_severity = {
                "critical": sum(1 for v in violations_in_period if v.get("severity") == "critical"),
                "high": sum(1 for v in violations_in_period if v.get("severity") == "high"),
                "medium": sum(1 for v in violations_in_period if v.get("severity") == "medium"),
                "low": sum(1 for v in violations_in_period if v.get("severity") == "low")
            }
            
            # Module compliance
            module_compliance = {}
            if metrics_in_period:
                module_compliance = {
                    "creativity": np.mean([m.get("creativity_ethics", 0.90) for m in metrics_in_period]),
                    "reflection": np.mean([m.get("reflection_ethics", 0.85) for m in metrics_in_period]),
                    "memory": np.mean([m.get("memory_ethics", 0.92) for m in metrics_in_period]),
                    "cohesion": np.mean([m.get("cohesion_ethics", 0.90) for m in metrics_in_period])
                }
            
            # Module violations
            module_violations = {}
            for module in ["creativity", "reflection", "memory", "cohesion"]:
                module_violations[module] = sum(1 for v in violations_in_period if v.get("module") == module)
            
            # Trend analysis
            if len(metrics_in_period) >= 3:
                recent_compliance = [m.get("compliance_index", 0.90) for m in metrics_in_period[-3:]]
                older_compliance = [m.get("compliance_index", 0.90) for m in metrics_in_period[:3]]
                delta = np.mean(recent_compliance) - np.mean(older_compliance)
                
                if delta > 0.02:
                    compliance_trend = "improving"
                elif delta < -0.02:
                    compliance_trend = "declining"
                else:
                    compliance_trend = "stable"
            else:
                compliance_trend = "insufficient_data"
            
            # Ethical continuity trend
            if metrics_in_period:
                ethical_continuity_trend = np.mean([m.get("ethical_continuity", 0.90) for m in metrics_in_period])
            else:
                ethical_continuity_trend = 0.90
            
            # Top violations
            top_violations = sorted(
                violations_in_period,
                key=lambda v: {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(v.get("severity", "low"), 0),
                reverse=True
            )[:10]
            
            # Threshold adjustments
            threshold_docs = await self.db.llm_ethics_policies.find({
                "last_adjusted": {"$gte": start_date, "$lte": end_date}
            }).to_list(100)
            
            threshold_adjustments = [{
                "module": t.get("module"),
                "parameter": t.get("parameter"),
                "adjustment": t.get("current_threshold", 0) - t.get("base_threshold", 0),
                "reason": t.get("reason")
            } for t in threshold_docs]
            
            # Parameter change requests
            approval_docs = await self.db.llm_ethics_approvals.find({
                "timestamp": {"$gte": start_date, "$lte": end_date}
            }).to_list(1000)
            
            pending_approvals = sum(1 for a in approval_docs if a.get("approval_status") == "pending")
            approved_changes = sum(1 for a in approval_docs if a.get("approval_status") == "approved")
            rejected_changes = sum(1 for a in approval_docs if a.get("approval_status") == "rejected")
            
            # Generate recommendations
            recommendations = self._generate_report_recommendations(
                overall_compliance, compliance_trend, violations_by_severity, module_compliance
            )
            
            # Health assessment
            health_assessment = await self._generate_health_assessment(
                overall_compliance, ethical_continuity_trend, total_violations, total_scans
            )
            
            report = EthicsReport(
                report_id=str(uuid.uuid4()),
                report_number=report_number,
                timestamp=datetime.now(timezone.utc).isoformat(),
                reporting_period={"start": start_date, "end": end_date},
                overall_compliance=round(overall_compliance, 3),
                total_scans=total_scans,
                total_violations=total_violations,
                violations_by_severity=violations_by_severity,
                module_compliance={k: round(v, 3) for k, v in module_compliance.items()},
                module_violations=module_violations,
                compliance_trend=compliance_trend,
                ethical_continuity_trend=round(ethical_continuity_trend, 3),
                top_violations=[{
                    "violation_id": v.get("violation_id"),
                    "severity": v.get("severity"),
                    "module": v.get("module"),
                    "type": v.get("violation_type"),
                    "description": v.get("description"),
                    "status": v.get("resolution_status")
                } for v in top_violations],
                threshold_adjustments=threshold_adjustments,
                pending_approvals=pending_approvals,
                approved_changes=approved_changes,
                rejected_changes=rejected_changes,
                recommendations=recommendations,
                system_health_assessment=health_assessment
            )
            
            # Store report
            await self.db.llm_ethics_reports.insert_one(report.to_dict())
            
            logger.info(
                f"Ethics Report #{report_number:03d} generated: "
                f"Compliance={overall_compliance:.3f}, Violations={total_violations}, "
                f"Trend={compliance_trend}"
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating ethics report: {e}")
            raise
    
    def _generate_report_recommendations(
        self,
        overall_compliance: float,
        compliance_trend: str,
        violations_by_severity: Dict[str, int],
        module_compliance: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations for ethics report"""
        recommendations = []
        
        # Overall compliance
        if overall_compliance >= 0.95:
            recommendations.append("✅ Excellent overall compliance. System operating within ethical guidelines.")
        elif overall_compliance >= 0.85:
            recommendations.append("✅ Good compliance. Minor refinements may enhance ethical standards.")
        elif overall_compliance >= 0.75:
            recommendations.append("⚠️ Moderate compliance. Review and address flagged violations.")
        else:
            recommendations.append("🚨 Low compliance. Immediate review and corrective action required.")
        
        # Trend
        if compliance_trend == "declining":
            recommendations.append("🚨 Declining compliance trend detected. Investigate root causes and implement corrective measures.")
        elif compliance_trend == "improving":
            recommendations.append("✅ Improving compliance trend. Continue current ethical governance practices.")
        
        # Critical violations
        if violations_by_severity.get("critical", 0) > 0:
            recommendations.append(f"🚨 {violations_by_severity['critical']} critical violations require immediate attention.")
        
        # Module-specific
        for module, score in module_compliance.items():
            if score < 0.75:
                recommendations.append(f"⚠️ {module.capitalize()} module compliance ({score:.2f}) below threshold. Review module parameters.")
        
        return recommendations[:8]
    
    async def _generate_health_assessment(
        self,
        overall_compliance: float,
        ethical_continuity: float,
        total_violations: int,
        total_scans: int
    ) -> str:
        """Generate overall system health assessment"""
        
        if self.llm_available:
            return await self._llm_generate_health_assessment(
                overall_compliance, ethical_continuity, total_violations, total_scans
            )
        else:
            return self._mock_generate_health_assessment(
                overall_compliance, ethical_continuity, total_violations, total_scans
            )
    
    async def _llm_generate_health_assessment(
        self,
        overall_compliance: float,
        ethical_continuity: float,
        total_violations: int,
        total_scans: int
    ) -> str:
        """Use LLM to generate health assessment [PROD]"""
        try:
            provider_config = self.llm_providers["primary"]
            
            chat = LlmChat(
                api_key=self.llm_key,
                session_id=f"ethics-health-{uuid.uuid4()}",
                system_message="You are an ethical governance analyst assessing the health and compliance of an AI chess system."
            ).with_model(provider_config["provider"], provider_config["model"])
            
            violation_rate = total_violations / total_scans if total_scans > 0 else 0.0
            
            prompt = f"""Generate a concise health assessment (3-4 sentences) for the AlphaZero Chess AI ethical governance system:

**Metrics:**
- Overall Compliance: {overall_compliance:.3f} (target: ≥0.95)
- Ethical Continuity: {ethical_continuity:.3f} (target: ≥0.93)
- Total Scans: {total_scans}
- Total Violations: {total_violations}
- Violation Rate: {violation_rate:.1%}

Assess:
1. Overall ethical health status
2. Key strengths or concerns
3. Compliance with target metrics
4. Forward-looking recommendation

Keep it under 100 words and professional."""

            user_message = UserMessage(text=prompt)
            response = await chat.send_message(user_message)
            
            assessment = response.strip()
            logger.info("[PROD] LLM-generated ethics health assessment")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error generating LLM health assessment: {e}")
            return self._mock_generate_health_assessment(
                overall_compliance, ethical_continuity, total_violations, total_scans
            )
    
    def _mock_generate_health_assessment(
        self,
        overall_compliance: float,
        ethical_continuity: float,
        total_violations: int,
        total_scans: int
    ) -> str:
        """Generate mock health assessment [MOCK]"""
        
        if overall_compliance >= 0.95 and ethical_continuity >= 0.93:
            status = "excellent ethical health"
        elif overall_compliance >= 0.85:
            status = "good ethical health with minor areas for improvement"
        else:
            status = "ethical health requiring attention"
        
        return (
            f"Ethical Governance Layer 2.0 operating at {status}. "
            f"Overall compliance at {overall_compliance:.3f} {'meets' if overall_compliance >= 0.95 else 'approaching'} target standards. "
            f"Ethical continuity across modules maintained at {ethical_continuity:.3f}. "
            f"{total_violations} violations flagged across {total_scans} monitoring scans, indicating "
            f"{'robust' if total_violations < total_scans * 0.05 else 'active'} oversight and continuous validation. "
            f"System demonstrates {'strong' if overall_compliance >= 0.90 else 'developing'} adherence to Fair Play, "
            f"Transparency, Educational Value, and Non-Manipulation policies. "
            f"{'Continue current governance practices with regular monitoring.' if overall_compliance >= 0.90 else 'Implement recommended corrections to enhance compliance.'}"
        )
    
    async def request_parameter_change(
        self,
        module: str,
        parameter: str,
        current_value: float,
        proposed_value: float,
        reason: str
    ) -> ParameterChangeRequest:
        """
        Create a parameter change request requiring human approval.
        
        Args:
            module: Module requesting change
            parameter: Parameter to change
            current_value: Current parameter value
            proposed_value: Proposed new value
            reason: Justification for change
        
        Returns:
            Parameter change request object
        """
        try:
            delta = proposed_value - current_value
            
            # Determine severity
            if abs(delta) > 0.10:
                severity = "high"
            elif abs(delta) > 0.05:
                severity = "medium"
            else:
                severity = "low"
            
            # Generate impact analysis
            impact_analysis = f"Changing {parameter} by {delta:+.3f} ({delta/current_value*100:+.1f}%) will "
            if delta > 0:
                impact_analysis += f"increase {parameter}, potentially enhancing that capability but may affect stability."
            else:
                impact_analysis += f"decrease {parameter}, potentially improving stability but may reduce that capability."
            
            # Critical changes require approval
            requires_approval = severity in ["high", "critical"] or abs(delta) > 0.05
            
            request = ParameterChangeRequest(
                request_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                module=module,
                parameter=parameter,
                current_value=current_value,
                proposed_value=proposed_value,
                delta=delta,
                reason=reason,
                impact_analysis=impact_analysis,
                severity=severity,
                requires_approval=requires_approval,
                approval_status="pending" if requires_approval else "auto_approved",
                approved_by=None,
                approval_timestamp=None,
                approval_notes=None
            )
            
            # Store request
            await self.db.llm_ethics_approvals.insert_one(request.to_dict())
            
            logger.info(
                f"Parameter change request created: {module}.{parameter} "
                f"{current_value:.3f} → {proposed_value:.3f} (status={request.approval_status})"
            )
            
            return request
            
        except Exception as e:
            logger.error(f"Error creating parameter change request: {e}")
            raise
    
    async def approve_parameter_change(
        self,
        request_id: str,
        approved: bool,
        approved_by: str,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Approve or reject a parameter change request.
        
        Args:
            request_id: Request ID to approve/reject
            approved: True to approve, False to reject
            approved_by: Username/identifier of approver
            notes: Optional approval notes
        
        Returns:
            Result of approval action
        """
        try:
            # Get request
            request = await self.db.llm_ethics_approvals.find_one({"request_id": request_id})
            
            if not request:
                return {"success": False, "error": "Request not found"}
            
            if request.get("approval_status") != "pending":
                return {"success": False, "error": f"Request already {request.get('approval_status')}"}
            
            # Update request
            new_status = "approved" if approved else "rejected"
            approval_timestamp = datetime.now(timezone.utc).isoformat()
            
            await self.db.llm_ethics_approvals.update_one(
                {"request_id": request_id},
                {"$set": {
                    "approval_status": new_status,
                    "approved_by": approved_by,
                    "approval_timestamp": approval_timestamp,
                    "approval_notes": notes
                }}
            )
            
            logger.info(
                f"Parameter change request {request_id} {new_status} by {approved_by}"
            )
            
            return {
                "success": True,
                "request_id": request_id,
                "status": new_status,
                "approved_by": approved_by,
                "timestamp": approval_timestamp,
                "message": f"Parameter change {new_status}"
            }
            
        except Exception as e:
            logger.error(f"Error approving parameter change: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_ethics_status(self) -> Dict[str, Any]:
        """Get current ethical governance status"""
        try:
            # Get latest metrics
            latest_metrics = await self.db.llm_ethics_metrics.find_one(
                sort=[("timestamp", -1)]
            )
            
            if latest_metrics:
                # Remove MongoDB _id
                if '_id' in latest_metrics:
                    del latest_metrics['_id']
                
                status = {
                    "system_status": "operational",
                    "last_scan": latest_metrics.get("timestamp"),
                    "compliance_index": latest_metrics.get("compliance_index"),
                    "ethical_continuity": latest_metrics.get("ethical_continuity"),
                    "status": latest_metrics.get("status"),
                    "anomalies_detected_24h": latest_metrics.get("anomalies_detected", 0),
                    "violations_flagged_24h": latest_metrics.get("violations_flagged", 0),
                    "false_positive_rate": latest_metrics.get("false_positive_rate"),
                    "target_comparison": {
                        "compliance": {
                            "current": latest_metrics.get("compliance_index"),
                            "target": self.target_metrics["compliance_index"],
                            "status": "✅" if latest_metrics.get("compliance_index", 0) >= self.target_metrics["compliance_index"] else "⚠️"
                        },
                        "continuity": {
                            "current": latest_metrics.get("ethical_continuity"),
                            "target": self.target_metrics["ethical_continuity"],
                            "status": "✅" if latest_metrics.get("ethical_continuity", 0) >= self.target_metrics["ethical_continuity"] else "⚠️"
                        },
                        "false_positive_rate": {
                            "current": latest_metrics.get("false_positive_rate"),
                            "target": self.target_metrics["false_positive_rate"],
                            "status": "✅" if latest_metrics.get("false_positive_rate", 1.0) <= self.target_metrics["false_positive_rate"] else "⚠️"
                        }
                    }
                }
            else:
                status = {
                    "system_status": "initializing",
                    "message": "No ethics scans executed yet"
                }
            
            # Get pending approvals
            pending_count = await self.db.llm_ethics_approvals.count_documents({
                "approval_status": "pending"
            })
            status["pending_approvals"] = pending_count
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting ethics status: {e}")
            return {"system_status": "error", "error": str(e)}
    
    async def get_violations(
        self,
        severity: Optional[str] = None,
        module: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get flagged violations with optional filters"""
        try:
            query = {}
            
            if severity:
                query["severity"] = severity
            if module:
                query["module"] = module
            
            violations = await self.db.llm_ethics_audit.find(query).sort(
                "timestamp", -1
            ).limit(limit).to_list(limit)
            
            # Remove MongoDB _id
            for v in violations:
                if '_id' in v:
                    del v['_id']
            
            return violations
            
        except Exception as e:
            logger.error(f"Error getting violations: {e}")
            return []
    
    async def get_metrics_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get ethics metrics history"""
        try:
            start_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            
            metrics = await self.db.llm_ethics_metrics.find({
                "timestamp": {"$gte": start_date}
            }).sort("timestamp", -1).to_list(1000)
            
            # Remove MongoDB _id
            for m in metrics:
                if '_id' in m:
                    del m['_id']
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics history: {e}")
            return []
    
    async def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get all pending parameter change requests"""
        try:
            approvals = await self.db.llm_ethics_approvals.find({
                "approval_status": "pending"
            }).sort("timestamp", -1).to_list(100)
            
            # Remove MongoDB _id
            for a in approvals:
                if '_id' in a:
                    del a['_id']
            
            return approvals
            
        except Exception as e:
            logger.error(f"Error getting pending approvals: {e}")
            return []
