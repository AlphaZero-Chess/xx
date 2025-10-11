"""
Adaptive Goal Formation & Ethical Governance Layer (Step 25)

This module implements autonomous goal formation with ethical constraint enforcement
and governance transparency for the AlphaZero Chess system.
"""

import logging
import uuid
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import numpy as np
import os

logger = logging.getLogger(__name__)


@dataclass
class GovernanceRule:
    """Ethical governance rule definition"""
    rule_id: str
    name: str
    description: str
    constraint_type: str  # "transparency", "fairness", "safety"
    threshold: float  # Numeric threshold (e.g., 0.10 for 10%)
    enabled: bool = True
    
    def to_dict(self):
        return asdict(self)


@dataclass
class AdaptiveGoal:
    """Adaptive goal with ethical alignment scoring"""
    goal_id: str
    timestamp: str
    goal_type: str  # "chess_performance", "ethical_alignment", "system_stability"
    description: str
    rationale: str
    strategic_alignment: float  # 0-1
    ethical_alignment: float  # 0-1
    stability_impact: float  # -1 to 1 (negative = destabilizing)
    confidence: float  # 0-1
    is_critical: bool  # Requires manual approval if True
    status: str  # "proposed", "approved", "rejected", "active", "completed"
    auto_apply: bool  # Can be auto-applied if True
    expected_outcomes: List[str]
    risks: List[str]
    approval_required: bool
    
    def to_dict(self):
        return asdict(self)


@dataclass
class GovernanceEvaluation:
    """Ethical evaluation of a proposed goal"""
    evaluation_id: str
    goal_id: str
    timestamp: str
    transparency_score: float  # 0-1
    fairness_score: float  # 0-1
    safety_score: float  # 0-1
    overall_alignment: float  # 0-1
    violations: List[str]
    recommendations: List[str]
    approval_status: str  # "approved", "rejected", "requires_review"
    
    def to_dict(self):
        return asdict(self)


class AdaptiveGoalController:
    """
    Controller for autonomous goal formation and ethical governance.
    Generates goals, evaluates alignment, and enforces governance rules.
    """
    
    def __init__(self, db_client):
        self.db = db_client
        self.llm_available = True
        
        # Default governance rules
        self.default_rules = [
            GovernanceRule(
                rule_id="rule_transparency_001",
                name="Transparency Preservation",
                description="Decisions must include an explainable summary",
                constraint_type="transparency",
                threshold=0.0,  # No tolerance for unexplainable decisions
                enabled=True
            ),
            GovernanceRule(
                rule_id="rule_fairness_001",
                name="Fairness Parity",
                description="No biased evaluations (parity checks required)",
                constraint_type="fairness",
                threshold=0.15,  # Max 15% bias variance allowed
                enabled=True
            ),
            GovernanceRule(
                rule_id="rule_safety_001",
                name="Interpretability Safety",
                description="Do not trade interpretability for >10% raw performance gain",
                constraint_type="safety",
                threshold=0.10,  # Max 10% performance gain vs interpretability loss
                enabled=True
            ),
            GovernanceRule(
                rule_id="rule_safety_002",
                name="Stability Preservation",
                description="Changes must not destabilize system by >±5%",
                constraint_type="safety",
                threshold=0.05,  # ±5% stability bounds
                enabled=True
            )
        ]
    
    async def generate_adaptive_goals(
        self, 
        performance_trends: Dict[str, Any],
        emergent_signals: Dict[str, Any]
    ) -> List[AdaptiveGoal]:
        """
        Generate new adaptive goals based on performance trends and emergent reasoning.
        
        Args:
            performance_trends: Recent system performance metrics
            emergent_signals: Signals from meta-reasoning system
        
        Returns:
            List of proposed adaptive goals
        """
        try:
            logger.info("Generating adaptive goals from performance trends...")
            
            goals = []
            
            # Analyze performance trends to derive goals
            goals.extend(await self._derive_performance_goals(performance_trends))
            
            # Analyze emergent signals for alignment goals
            goals.extend(await self._derive_alignment_goals(emergent_signals))
            
            # Generate stability goals if needed
            goals.extend(await self._derive_stability_goals(performance_trends, emergent_signals))
            
            logger.info(f"Generated {len(goals)} adaptive goals")
            return goals
            
        except Exception as e:
            logger.error(f"Error generating adaptive goals: {e}")
            return []
    
    async def _derive_performance_goals(self, trends: Dict[str, Any]) -> List[AdaptiveGoal]:
        """Derive chess performance improvement goals"""
        goals = []
        
        try:
            # Check if we have LLM available
            if self.llm_available:
                llm_goals = await self._llm_generate_performance_goals(trends)
                goals.extend(llm_goals)
            else:
                # Fallback: Rule-based goal generation
                goals.extend(self._fallback_performance_goals(trends))
            
            return goals
            
        except Exception as e:
            logger.error(f"Error deriving performance goals: {e}")
            return self._fallback_performance_goals(trends)
    
    async def _llm_generate_performance_goals(self, trends: Dict[str, Any]) -> List[AdaptiveGoal]:
        """Use LLM to generate performance goals"""
        try:
            from emergentintegrations import OpenAI
            
            # Get Emergent LLM key
            llm_key = os.environ.get('EMERGENT_LLM_KEY')
            if not llm_key:
                logger.warning("EMERGENT_LLM_KEY not found, using fallback")
                return []
            
            client = OpenAI(api_key=llm_key)
            
            prompt = f"""Analyze the following AlphaZero Chess system performance trends and propose 2-3 adaptive goals for improvement:

Performance Trends:
{self._format_trends(trends)}

Generate goals that:
1. Improve chess playing strength
2. Maintain ethical alignment (transparency, fairness, safety)
3. Balance performance vs interpretability
4. Are measurable and achievable

For each goal, provide:
- Goal type (chess_performance, ethical_alignment, or system_stability)
- Description (1 sentence)
- Rationale (2 sentences)
- Expected outcomes (2-3 items)
- Potential risks (1-2 items)
- Whether it requires manual approval (critical: yes/no)

Format as JSON array."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI governance expert for AlphaZero Chess. Generate adaptive goals that balance performance and ethics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            # Parse LLM response
            llm_text = response.choices[0].message.content.strip()
            goals = self._parse_llm_goals(llm_text, trends)
            
            logger.info(f"LLM generated {len(goals)} performance goals")
            return goals
            
        except Exception as e:
            logger.error(f"Error in LLM goal generation: {e}")
            self.llm_available = False
            return []
    
    def _format_trends(self, trends: Dict[str, Any]) -> str:
        """Format trends for LLM prompt"""
        lines = []
        for key, value in trends.items():
            if isinstance(value, (int, float)):
                lines.append(f"- {key}: {value}")
            elif isinstance(value, dict):
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)
    
    def _parse_llm_goals(self, llm_text: str, trends: Dict[str, Any]) -> List[AdaptiveGoal]:
        """Parse LLM response into AdaptiveGoal objects"""
        goals = []
        
        try:
            import json
            
            # Try to extract JSON from response
            start_idx = llm_text.find('[')
            end_idx = llm_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = llm_text[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                for item in parsed:
                    goal = AdaptiveGoal(
                        goal_id=str(uuid.uuid4()),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        goal_type=item.get('goal_type', 'chess_performance'),
                        description=item.get('description', ''),
                        rationale=item.get('rationale', ''),
                        strategic_alignment=0.8,  # Will be evaluated
                        ethical_alignment=0.8,  # Will be evaluated
                        stability_impact=0.0,  # Will be evaluated
                        confidence=0.75,
                        is_critical=item.get('critical', 'no').lower() == 'yes',
                        status='proposed',
                        auto_apply=False,
                        expected_outcomes=item.get('expected_outcomes', []),
                        risks=item.get('risks', []),
                        approval_required=item.get('critical', 'no').lower() == 'yes'
                    )
                    goals.append(goal)
            
        except Exception as e:
            logger.error(f"Error parsing LLM goals: {e}")
        
        return goals
    
    def _fallback_performance_goals(self, trends: Dict[str, Any]) -> List[AdaptiveGoal]:
        """Fallback rule-based goal generation (MOCKED)"""
        goals = []
        
        # Mock goal 1: Improve MCTS simulations
        goals.append(AdaptiveGoal(
            goal_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            goal_type="chess_performance",
            description="Increase MCTS simulations to 900 for improved tactical accuracy",
            rationale="Current simulation count may be limiting search depth. Analysis shows 12% of positions require deeper search. **[MOCKED]**",
            strategic_alignment=0.85,
            ethical_alignment=0.90,
            stability_impact=0.03,
            confidence=0.78,
            is_critical=False,
            status='proposed',
            auto_apply=True,
            expected_outcomes=[
                "Improved position evaluation accuracy",
                "Better endgame performance",
                "Slight increase in computation time"
            ],
            risks=[
                "Increased latency per move",
                "Minimal stability impact"
            ],
            approval_required=False
        ))
        
        logger.info(f"**FALLBACK MODE**: Generated {len(goals)} mocked goals (LLM unavailable)")
        return goals
    
    async def _derive_alignment_goals(self, signals: Dict[str, Any]) -> List[AdaptiveGoal]:
        """Derive ethical alignment goals"""
        goals = []
        
        # Check alignment metrics
        alignment_score = signals.get('overall_alignment', 0.85)
        
        if alignment_score < 0.80:
            goals.append(AdaptiveGoal(
                goal_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                goal_type="ethical_alignment",
                description="Increase trust calibration threshold to improve consensus alignment",
                rationale=f"Current alignment at {alignment_score:.1%} below target 80%. Higher trust threshold will reduce low-confidence decisions.",
                strategic_alignment=0.70,
                ethical_alignment=0.95,
                stability_impact=-0.02,
                confidence=0.82,
                is_critical=False,
                status='proposed',
                auto_apply=True,
                expected_outcomes=[
                    "Improved collective alignment",
                    "Reduced arbitration conflicts",
                    "Higher consensus stability"
                ],
                risks=[
                    "May reduce decision throughput slightly"
                ],
                approval_required=False
            ))
        
        return goals
    
    async def _derive_stability_goals(
        self, 
        trends: Dict[str, Any], 
        signals: Dict[str, Any]
    ) -> List[AdaptiveGoal]:
        """Derive system stability goals"""
        goals = []
        
        # Check trust variance
        trust_variance = signals.get('trust_variance', 0.08)
        
        if trust_variance > 0.15:
            goals.append(AdaptiveGoal(
                goal_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                goal_type="system_stability",
                description="Reduce trust variance by adjusting arbitration thresholds",
                rationale=f"Trust variance at {trust_variance:.2f} exceeds 0.15 threshold. System showing instability in consensus formation.",
                strategic_alignment=0.75,
                ethical_alignment=0.88,
                stability_impact=0.08,
                confidence=0.80,
                is_critical=True,  # High variance is critical
                status='proposed',
                auto_apply=False,
                expected_outcomes=[
                    "Reduced trust variance",
                    "More stable consensus",
                    "Consistent decision patterns"
                ],
                risks=[
                    "May slow adaptation rate",
                    "Could mask emerging patterns"
                ],
                approval_required=True
            ))
        
        return goals
    
    async def evaluate_goal_alignment(self, goal: AdaptiveGoal) -> GovernanceEvaluation:
        """
        Evaluate a goal against ethical governance rules.
        
        Args:
            goal: The adaptive goal to evaluate
        
        Returns:
            GovernanceEvaluation with alignment scores and recommendations
        """
        try:
            evaluation_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Get active governance rules
            rules = await self._get_governance_rules()
            
            # Evaluate transparency
            transparency_score = self._evaluate_transparency(goal)
            
            # Evaluate fairness
            fairness_score = self._evaluate_fairness(goal)
            
            # Evaluate safety
            safety_score, safety_violations = self._evaluate_safety(goal, rules)
            
            # Check for violations
            violations = []
            violations.extend(safety_violations)
            
            if transparency_score < 0.7:
                violations.append("Low transparency: Goal lacks clear explainability")
            
            if fairness_score < 0.7:
                violations.append("Fairness concern: Potential bias detected")
            
            # Calculate overall alignment
            overall_alignment = (transparency_score + fairness_score + safety_score) / 3.0
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                goal, transparency_score, fairness_score, safety_score, violations
            )
            
            # Determine approval status
            if violations:
                if goal.is_critical or overall_alignment < 0.65:
                    approval_status = "rejected"
                else:
                    approval_status = "requires_review"
            else:
                if overall_alignment >= 0.75 and not goal.is_critical:
                    approval_status = "approved"
                else:
                    approval_status = "requires_review"
            
            evaluation = GovernanceEvaluation(
                evaluation_id=evaluation_id,
                goal_id=goal.goal_id,
                timestamp=timestamp,
                transparency_score=transparency_score,
                fairness_score=fairness_score,
                safety_score=safety_score,
                overall_alignment=overall_alignment,
                violations=violations,
                recommendations=recommendations,
                approval_status=approval_status
            )
            
            logger.info(f"Goal {goal.goal_id[:8]} evaluated: {approval_status} (alignment: {overall_alignment:.2f})")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating goal alignment: {e}")
            # Return conservative evaluation
            return GovernanceEvaluation(
                evaluation_id=str(uuid.uuid4()),
                goal_id=goal.goal_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                transparency_score=0.5,
                fairness_score=0.5,
                safety_score=0.5,
                overall_alignment=0.5,
                violations=["Evaluation error occurred"],
                recommendations=["Manual review required"],
                approval_status="requires_review"
            )
    
    async def _get_governance_rules(self) -> List[GovernanceRule]:
        """Get active governance rules from database or defaults"""
        try:
            rules_docs = await self.db.llm_governance_rules.find({"enabled": True}).to_list(100)
            
            if rules_docs:
                # Remove MongoDB _id field before creating GovernanceRule objects
                rules = []
                for doc in rules_docs:
                    doc.pop('_id', None)  # Remove _id if present
                    rules.append(GovernanceRule(**doc))
                return rules
            else:
                # Initialize with defaults
                for rule in self.default_rules:
                    await self.db.llm_governance_rules.insert_one(rule.to_dict())
                return self.default_rules
                
        except Exception as e:
            logger.error(f"Error getting governance rules: {e}")
            return self.default_rules
    
    def _evaluate_transparency(self, goal: AdaptiveGoal) -> float:
        """Evaluate transparency score (0-1)"""
        score = 0.8  # Base score
        
        # Check if description is clear
        if len(goal.description) < 20:
            score -= 0.2
        
        # Check if rationale is provided
        if len(goal.rationale) < 30:
            score -= 0.15
        
        # Check if expected outcomes are listed
        if len(goal.expected_outcomes) == 0:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_fairness(self, goal: AdaptiveGoal) -> float:
        """Evaluate fairness score (0-1)"""
        score = 0.85  # Base score
        
        # Check if goal might introduce bias
        bias_keywords = ['favor', 'bias', 'prefer', 'discriminate']
        text = (goal.description + " " + goal.rationale).lower()
        
        for keyword in bias_keywords:
            if keyword in text:
                score -= 0.15
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_safety(
        self, 
        goal: AdaptiveGoal, 
        rules: List[GovernanceRule]
    ) -> Tuple[float, List[str]]:
        """Evaluate safety score and detect violations"""
        score = 0.9
        violations = []
        
        # Check stability impact against rules
        for rule in rules:
            if rule.constraint_type == "safety" and rule.enabled:
                if "stability" in rule.name.lower():
                    if abs(goal.stability_impact) > rule.threshold:
                        violations.append(
                            f"Stability impact {goal.stability_impact:.2%} exceeds ±{rule.threshold:.0%} threshold"
                        )
                        score -= 0.3
                
                if "interpretability" in rule.description.lower():
                    # Check if goal trades interpretability for performance
                    if goal.goal_type == "chess_performance" and abs(goal.stability_impact) > rule.threshold:
                        violations.append(
                            f"Performance gain may exceed {rule.threshold:.0%} interpretability limit"
                        )
                        score -= 0.2
        
        return max(0.0, min(1.0, score)), violations
    
    def _generate_recommendations(
        self,
        goal: AdaptiveGoal,
        transparency: float,
        fairness: float,
        safety: float,
        violations: List[str]
    ) -> List[str]:
        """Generate recommendations based on evaluation"""
        recommendations = []
        
        if transparency < 0.75:
            recommendations.append("Improve goal description and rationale clarity")
        
        if fairness < 0.75:
            recommendations.append("Review for potential bias or unfair advantage")
        
        if safety < 0.75:
            recommendations.append("Reduce stability impact or provide risk mitigation")
        
        if goal.is_critical:
            recommendations.append("Critical goal requires manual approval before execution")
        
        if not violations and transparency >= 0.8 and fairness >= 0.8 and safety >= 0.8:
            recommendations.append("Goal meets all governance criteria - safe to approve")
        
        return recommendations
    
    async def apply_governance_rules(
        self, 
        goal: AdaptiveGoal, 
        evaluation: GovernanceEvaluation
    ) -> Tuple[bool, str]:
        """
        Apply governance rules to determine if goal can be executed.
        
        Args:
            goal: The adaptive goal
            evaluation: Ethical evaluation
        
        Returns:
            Tuple of (can_execute, reason)
        """
        try:
            # Check evaluation status
            if evaluation.approval_status == "rejected":
                return False, f"Goal rejected due to governance violations: {', '.join(evaluation.violations)}"
            
            # Check if critical (always requires manual approval)
            if goal.is_critical:
                return False, "Critical goal requires manual approval"
            
            # Check if auto-apply is allowed
            if not goal.auto_apply:
                return False, "Goal requires manual approval (auto_apply=False)"
            
            # Check alignment threshold
            if evaluation.overall_alignment < 0.75:
                return False, f"Overall alignment {evaluation.overall_alignment:.2f} below 0.75 threshold"
            
            # Check stability bounds (±5%)
            if abs(goal.stability_impact) > 0.05:
                return False, f"Stability impact {goal.stability_impact:.2%} exceeds ±5% safety bounds"
            
            # All checks passed
            return True, "Goal approved for automatic execution"
            
        except Exception as e:
            logger.error(f"Error applying governance rules: {e}")
            return False, f"Error in governance evaluation: {str(e)}"
    
    async def record_goal_outcome(
        self, 
        goal: AdaptiveGoal, 
        evaluation: GovernanceEvaluation,
        executed: bool,
        outcome: Dict[str, Any]
    ) -> None:
        """
        Record goal execution outcome in governance log.
        
        Args:
            goal: The adaptive goal
            evaluation: Ethical evaluation
            executed: Whether goal was executed
            outcome: Execution results
        """
        try:
            log_entry = {
                "log_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "goal_id": goal.goal_id,
                "goal_type": goal.goal_type,
                "goal_description": goal.description,
                "evaluation_id": evaluation.evaluation_id,
                "approval_status": evaluation.approval_status,
                "overall_alignment": evaluation.overall_alignment,
                "transparency_score": evaluation.transparency_score,
                "fairness_score": evaluation.fairness_score,
                "safety_score": evaluation.safety_score,
                "violations": evaluation.violations,
                "executed": executed,
                "outcome": outcome,
                "is_critical": goal.is_critical,
                "auto_applied": goal.auto_apply and executed
            }
            
            await self.db.llm_governance_log.insert_one(log_entry)
            logger.info(f"Recorded governance log for goal {goal.goal_id[:8]}: executed={executed}")
            
        except Exception as e:
            logger.error(f"Error recording goal outcome: {e}")
    
    async def get_active_goals(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get list of active and proposed goals"""
        try:
            goals = await self.db.llm_adaptive_goals.find(
                {"status": {"$in": ["proposed", "approved", "active"]}}
            ).sort("timestamp", -1).limit(limit).to_list(limit)
            
            return goals
            
        except Exception as e:
            logger.error(f"Error getting active goals: {e}")
            return []
    
    async def approve_goal(self, goal_id: str, approved: bool, approver: str = "manual") -> bool:
        """Manually approve or reject a goal"""
        try:
            new_status = "approved" if approved else "rejected"
            
            result = await self.db.llm_adaptive_goals.update_one(
                {"goal_id": goal_id},
                {"$set": {
                    "status": new_status,
                    "approval_timestamp": datetime.now(timezone.utc).isoformat(),
                    "approver": approver
                }}
            )
            
            if result.modified_count > 0:
                logger.info(f"Goal {goal_id[:8]} {new_status} by {approver}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error approving goal: {e}")
            return False
    
    async def generate_governance_report(self) -> Dict[str, Any]:
        """Generate comprehensive governance report"""
        try:
            # Get governance logs
            total_logs = await self.db.llm_governance_log.count_documents({})
            
            # Get recent logs
            recent_logs = await self.db.llm_governance_log.find().sort(
                "timestamp", -1
            ).limit(50).to_list(50)
            
            # Calculate statistics
            if recent_logs:
                executed_count = sum(1 for log in recent_logs if log.get("executed", False))
                auto_applied_count = sum(1 for log in recent_logs if log.get("auto_applied", False))
                
                avg_alignment = np.mean([log.get("overall_alignment", 0) for log in recent_logs])
                avg_transparency = np.mean([log.get("transparency_score", 0) for log in recent_logs])
                avg_fairness = np.mean([log.get("fairness_score", 0) for log in recent_logs])
                avg_safety = np.mean([log.get("safety_score", 0) for log in recent_logs])
                
                violation_count = sum(len(log.get("violations", [])) for log in recent_logs)
                
                # Goal type distribution
                goal_types = {}
                for log in recent_logs:
                    gtype = log.get("goal_type", "unknown")
                    goal_types[gtype] = goal_types.get(gtype, 0) + 1
                
                # Approval status distribution
                approval_statuses = {}
                for log in recent_logs:
                    status = log.get("approval_status", "unknown")
                    approval_statuses[status] = approval_statuses.get(status, 0) + 1
            else:
                executed_count = 0
                auto_applied_count = 0
                avg_alignment = 0.0
                avg_transparency = 0.0
                avg_fairness = 0.0
                avg_safety = 0.0
                violation_count = 0
                goal_types = {}
                approval_statuses = {}
            
            # Get active governance rules
            rules = await self._get_governance_rules()
            
            report = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_governance_logs": total_logs,
                "recent_period_stats": {
                    "logs_analyzed": len(recent_logs),
                    "goals_executed": executed_count,
                    "goals_auto_applied": auto_applied_count,
                    "execution_rate": executed_count / len(recent_logs) if recent_logs else 0.0
                },
                "alignment_metrics": {
                    "overall_alignment": round(avg_alignment, 3),
                    "transparency_score": round(avg_transparency, 3),
                    "fairness_score": round(avg_fairness, 3),
                    "safety_score": round(avg_safety, 3)
                },
                "violations": {
                    "total_violations": violation_count,
                    "violation_rate": violation_count / len(recent_logs) if recent_logs else 0.0
                },
                "goal_distribution": goal_types,
                "approval_distribution": approval_statuses,
                "active_governance_rules": [rule.to_dict() for rule in rules],
                "system_health": self._calculate_governance_health(
                    avg_alignment, violation_count, len(recent_logs)
                )
            }
            
            logger.info("Generated governance report")
            return report
            
        except Exception as e:
            logger.error(f"Error generating governance report: {e}")
            return {
                "error": str(e),
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
    
    def _calculate_governance_health(
        self, 
        avg_alignment: float, 
        violation_count: int, 
        total_logs: int
    ) -> Dict[str, Any]:
        """Calculate governance system health"""
        
        # Health score based on alignment and violation rate
        violation_rate = violation_count / total_logs if total_logs > 0 else 0.0
        
        health_score = (avg_alignment * 0.7 + (1.0 - violation_rate) * 0.3) * 100
        
        if health_score >= 85:
            status = "excellent"
        elif health_score >= 70:
            status = "good"
        elif health_score >= 50:
            status = "needs_attention"
        else:
            status = "critical"
        
        return {
            "health_score": round(health_score, 1),
            "status": status,
            "avg_alignment": round(avg_alignment, 3),
            "violation_rate": round(violation_rate, 3)
        }
