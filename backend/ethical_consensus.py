"""
Emergent Ethical Consensus & System-Level Reasoning (Step 26)

This module implements multi-agent ethical deliberation, consensus voting,
conflict resolution, and adaptive governance rule refinement for AlphaZero Chess.
"""

import logging
import uuid
import asyncio
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from dotenv import load_dotenv

# Import emergentintegrations for multi-LLM support
from emergentintegrations.llm.chat import LlmChat, UserMessage

logger = logging.getLogger(__name__)
load_dotenv()


class AgentProvider(Enum):
    """LLM provider types for ethical agents"""
    OPENAI_GPT5 = ("openai", "gpt-5")
    ANTHROPIC_CLAUDE4 = ("anthropic", "claude-4-sonnet-20250514")
    GEMINI_25_PRO = ("gemini", "gemini-2.5-pro")


@dataclass
class AgentOpinion:
    """Individual agent's ethical opinion"""
    agent_id: str
    agent_name: str
    provider: str
    model: str
    opinion: str
    reasoning: str
    alignment_score: float  # 0-1
    confidence: float  # 0-1
    vote: str  # "approve", "reject", "abstain"
    timestamp: str
    response_time: float
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ConsensusResult:
    """Result of multi-agent consensus deliberation"""
    consensus_id: str
    timestamp: str
    goal_id: str
    goal_description: str
    total_agents: int
    agents_participated: int
    agent_opinions: List[Dict[str, Any]]
    vote_distribution: Dict[str, int]  # {"approve": 3, "reject": 1, "abstain": 1}
    consensus_reached: bool
    consensus_threshold: float
    agreement_score: float  # 0-1 (Ethical Alignment Index)
    agreement_variance: float  # σ for consensus stability
    final_decision: str  # "approved", "rejected", "requires_review"
    reasoning_summary: str
    conflicts_detected: List[str]
    conflict_resolution: Optional[str]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class RuleRefinement:
    """Governance rule refinement record"""
    refinement_id: str
    timestamp: str
    rule_id: str
    rule_name: str
    previous_weight: float
    new_weight: float
    weight_delta: float
    refinement_reason: str
    consensus_id: Optional[str]
    auto_applied: bool
    requires_approval: bool
    
    def to_dict(self):
        return asdict(self)


class EthicalConsensusEngine:
    """
    Multi-agent ethical consensus engine for distributed deliberation.
    
    Implements:
    - Multi-agent LLM deliberation (GPT-5, Claude 4, Gemini 2.5)
    - Weighted consensus voting
    - Conflict detection and resolution
    - Adaptive governance rule refinement
    - Transparency reporting
    """
    
    def __init__(self, db_client):
        self.db = db_client
        self.llm_key = os.environ.get('EMERGENT_LLM_KEY')
        
        if not self.llm_key:
            logger.error("EMERGENT_LLM_KEY not found in environment!")
            self.agents_available = False
        else:
            self.agents_available = True
            logger.info("Ethical Consensus Engine initialized with multi-LLM support")
        
        # Agent configuration: 5 agents with diverse perspectives
        self.agent_configs = [
            {
                "agent_id": "agent_gpt5_primary",
                "name": "GPT-5 Primary Ethicist",
                "provider": "openai",
                "model": "gpt-5",
                "weight": 1.2,  # Higher weight for primary agent
                "specialty": "comprehensive ethical analysis"
            },
            {
                "agent_id": "agent_gpt5_secondary",
                "name": "GPT-5 Secondary Analyst",
                "provider": "openai",
                "model": "gpt-5",
                "weight": 1.0,
                "specialty": "risk assessment and safety"
            },
            {
                "agent_id": "agent_claude4",
                "name": "Claude 4 Sonnet Ethicist",
                "provider": "anthropic",
                "model": "claude-4-sonnet-20250514",
                "weight": 1.1,
                "specialty": "nuanced reasoning and transparency"
            },
            {
                "agent_id": "agent_gemini_primary",
                "name": "Gemini 2.5 Pro Primary",
                "provider": "gemini",
                "model": "gemini-2.5-pro",
                "weight": 1.0,
                "specialty": "alignment and fairness analysis"
            },
            {
                "agent_id": "agent_gemini_secondary",
                "name": "Gemini 2.5 Pro Secondary",
                "provider": "gemini",
                "model": "gemini-2.5-pro",
                "weight": 0.9,
                "specialty": "stability and consistency checks"
            }
        ]
        
        # Base consensus threshold (adaptive)
        self.base_threshold = 0.70
    
    async def aggregate_agent_ethics(
        self,
        goal: Dict[str, Any],
        governance_context: Dict[str, Any]
    ) -> List[AgentOpinion]:
        """
        Gather ethical judgments from all configured LLM agents.
        
        Args:
            goal: The adaptive goal to evaluate
            governance_context: Current governance rules and metrics
        
        Returns:
            List of agent opinions
        """
        try:
            if not self.agents_available:
                logger.error("LLM agents not available - cannot aggregate ethics")
                return []
            
            logger.info(f"Aggregating ethical judgments from {len(self.agent_configs)} agents...")
            
            # Create tasks for parallel agent queries
            tasks = []
            for agent_config in self.agent_configs:
                task = self._query_agent(agent_config, goal, governance_context)
                tasks.append(task)
            
            # Execute all agent queries in parallel
            agent_opinions = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out failed queries
            valid_opinions = []
            for opinion in agent_opinions:
                if isinstance(opinion, AgentOpinion):
                    valid_opinions.append(opinion)
                elif isinstance(opinion, Exception):
                    logger.error(f"Agent query failed: {opinion}")
            
            logger.info(f"Received {len(valid_opinions)} valid agent opinions")
            return valid_opinions
            
        except Exception as e:
            logger.error(f"Error aggregating agent ethics: {e}")
            return []
    
    async def _query_agent(
        self,
        agent_config: Dict[str, Any],
        goal: Dict[str, Any],
        governance_context: Dict[str, Any]
    ) -> AgentOpinion:
        """Query a single LLM agent for ethical opinion"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Initialize LLM chat for this agent
            chat = LlmChat(
                api_key=self.llm_key,
                session_id=f"ethics-{agent_config['agent_id']}-{uuid.uuid4()}",
                system_message=self._build_agent_system_prompt(agent_config)
            ).with_model(agent_config['provider'], agent_config['model'])
            
            # Build the ethical evaluation prompt
            prompt = self._build_ethical_prompt(goal, governance_context, agent_config)
            
            # Query the agent
            user_message = UserMessage(text=prompt)
            response = await chat.send_message(user_message)
            
            elapsed = asyncio.get_event_loop().time() - start_time
            
            # Parse response
            opinion_data = self._parse_agent_response(response, agent_config)
            
            # Create AgentOpinion object
            opinion = AgentOpinion(
                agent_id=agent_config['agent_id'],
                agent_name=agent_config['name'],
                provider=agent_config['provider'],
                model=agent_config['model'],
                opinion=opinion_data['opinion'],
                reasoning=opinion_data['reasoning'],
                alignment_score=opinion_data['alignment_score'],
                confidence=opinion_data['confidence'],
                vote=opinion_data['vote'],
                timestamp=datetime.now(timezone.utc).isoformat(),
                response_time=elapsed
            )
            
            logger.info(f"{agent_config['name']}: {opinion.vote} (alignment: {opinion.alignment_score:.2f})")
            return opinion
            
        except Exception as e:
            logger.error(f"Error querying agent {agent_config['agent_id']}: {e}")
            raise
    
    def _build_agent_system_prompt(self, agent_config: Dict[str, Any]) -> str:
        """Build system prompt for ethical agent"""
        return f"""You are {agent_config['name']}, an ethical AI advisor specializing in {agent_config['specialty']}.

Your role is to evaluate proposed adaptive goals for an AlphaZero Chess AI system from an ethical perspective.

Focus areas:
- Transparency: Can the goal be clearly explained and understood?
- Fairness: Does the goal avoid bias and maintain fairness?
- Safety: Does the goal preserve system stability and interpretability?
- Alignment: Does the goal align with human values and expectations?

Provide thoughtful, nuanced analysis considering both benefits and risks."""
    
    def _build_ethical_prompt(
        self,
        goal: Dict[str, Any],
        governance_context: Dict[str, Any],
        agent_config: Dict[str, Any]
    ) -> str:
        """Build the ethical evaluation prompt"""
        
        rules_summary = "\n".join([
            f"- {rule['name']}: {rule['description']} (threshold: {rule['threshold']})"
            for rule in governance_context.get('active_rules', [])[:5]
        ])
        
        prompt = f"""Evaluate the following adaptive goal for an AlphaZero Chess AI system:

**Goal Information:**
- Type: {goal.get('goal_type', 'unknown')}
- Description: {goal.get('description', '')}
- Rationale: {goal.get('rationale', '')}
- Expected Outcomes: {', '.join(goal.get('expected_outcomes', []))}
- Potential Risks: {', '.join(goal.get('risks', []))}
- Is Critical: {goal.get('is_critical', False)}
- Stability Impact: {goal.get('stability_impact', 0):.2%}

**Governance Context:**
Active Governance Rules:
{rules_summary}

Current Metrics:
- Average Alignment: {governance_context.get('avg_alignment', 0):.2%}
- Recent Violations: {governance_context.get('recent_violations', 0)}

**Your Task:**
Provide your ethical assessment with:
1. Your overall opinion (2-3 sentences)
2. Detailed reasoning considering transparency, fairness, and safety
3. Alignment score (0.0 to 1.0)
4. Confidence in your assessment (0.0 to 1.0)
5. Your vote: "approve", "reject", or "abstain"

Format your response as JSON:
{{
  "opinion": "brief opinion",
  "reasoning": "detailed reasoning",
  "alignment_score": 0.85,
  "confidence": 0.90,
  "vote": "approve"
}}"""
        
        return prompt
    
    def _parse_agent_response(
        self,
        response: str,
        agent_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse agent's JSON response"""
        try:
            import json
            
            # Try to extract JSON from response
            response_text = response.strip()
            
            # Find JSON boundaries
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                return {
                    'opinion': parsed.get('opinion', 'No opinion provided'),
                    'reasoning': parsed.get('reasoning', 'No reasoning provided'),
                    'alignment_score': float(parsed.get('alignment_score', 0.5)),
                    'confidence': float(parsed.get('confidence', 0.5)),
                    'vote': parsed.get('vote', 'abstain').lower()
                }
            else:
                # Fallback parsing
                logger.warning(f"Could not extract JSON from {agent_config['name']} response")
                return self._fallback_parse(response_text)
                
        except Exception as e:
            logger.error(f"Error parsing agent response: {e}")
            return self._fallback_parse(response)
    
    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """Fallback parsing when JSON extraction fails"""
        # Simple heuristic parsing
        vote = "abstain"
        if "approve" in response.lower() and "not approve" not in response.lower():
            vote = "approve"
        elif "reject" in response.lower():
            vote = "reject"
        
        return {
            'opinion': response[:200] if len(response) > 200 else response,
            'reasoning': response,
            'alignment_score': 0.5,
            'confidence': 0.5,
            'vote': vote
        }
    
    async def run_consensus_voting(
        self,
        agent_opinions: List[AgentOpinion],
        goal: Dict[str, Any]
    ) -> ConsensusResult:
        """
        Run weighted voting and consensus analysis.
        
        Args:
            agent_opinions: List of agent opinions
            goal: The goal being evaluated
        
        Returns:
            ConsensusResult with voting outcomes and metrics
        """
        try:
            consensus_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()
            
            if not agent_opinions:
                logger.error("No agent opinions to process")
                return self._create_empty_consensus(consensus_id, timestamp, goal)
            
            # Calculate vote distribution
            vote_counts = {"approve": 0, "reject": 0, "abstain": 0}
            weighted_votes = {"approve": 0.0, "reject": 0.0, "abstain": 0.0}
            
            alignment_scores = []
            
            for opinion in agent_opinions:
                vote = opinion.vote.lower()
                if vote in vote_counts:
                    vote_counts[vote] += 1
                    
                    # Find agent weight
                    agent_weight = next(
                        (a['weight'] for a in self.agent_configs if a['agent_id'] == opinion.agent_id),
                        1.0
                    )
                    weighted_votes[vote] += agent_weight
                    alignment_scores.append(opinion.alignment_score)
            
            # Calculate consensus metrics
            total_weight = sum(weighted_votes.values())
            approval_ratio = weighted_votes["approve"] / total_weight if total_weight > 0 else 0.0
            
            # Calculate Ethical Alignment Index (EAI)
            agreement_score = np.mean(alignment_scores) if alignment_scores else 0.0
            
            # Calculate Agreement Variance (σ)
            agreement_variance = np.std(alignment_scores) if len(alignment_scores) > 1 else 0.0
            
            # Determine adaptive threshold based on goal criticality
            threshold = self._calculate_adaptive_threshold(goal)
            
            # Check if consensus reached
            consensus_reached = approval_ratio >= threshold and agreement_variance < 0.20
            
            # Determine final decision
            if consensus_reached and vote_counts["approve"] > vote_counts["reject"]:
                final_decision = "approved"
            elif vote_counts["reject"] > vote_counts["approve"]:
                final_decision = "rejected"
            else:
                final_decision = "requires_review"
            
            # Detect conflicts
            conflicts = self._detect_conflicts(agent_opinions, vote_counts, agreement_variance)
            
            # Generate reasoning summary
            reasoning_summary = self._generate_reasoning_summary(
                agent_opinions, vote_counts, agreement_score, consensus_reached
            )
            
            consensus_result = ConsensusResult(
                consensus_id=consensus_id,
                timestamp=timestamp,
                goal_id=goal.get('goal_id', 'unknown'),
                goal_description=goal.get('description', ''),
                total_agents=len(self.agent_configs),
                agents_participated=len(agent_opinions),
                agent_opinions=[op.to_dict() for op in agent_opinions],
                vote_distribution=vote_counts,
                consensus_reached=consensus_reached,
                consensus_threshold=threshold,
                agreement_score=agreement_score,
                agreement_variance=agreement_variance,
                final_decision=final_decision,
                reasoning_summary=reasoning_summary,
                conflicts_detected=conflicts,
                conflict_resolution=None
            )
            
            logger.info(
                f"Consensus: {final_decision} "
                f"(approval: {approval_ratio:.1%}, EAI: {agreement_score:.2f}, σ: {agreement_variance:.2f})"
            )
            
            return consensus_result
            
        except Exception as e:
            logger.error(f"Error in consensus voting: {e}")
            return self._create_empty_consensus(str(uuid.uuid4()), datetime.now(timezone.utc).isoformat(), goal)
    
    def _calculate_adaptive_threshold(self, goal: Dict[str, Any]) -> float:
        """Calculate adaptive consensus threshold based on goal criticality"""
        
        base_threshold = self.base_threshold  # 0.70
        
        # Increase threshold for critical goals
        if goal.get('is_critical', False):
            return min(0.90, base_threshold + 0.20)
        
        # Increase threshold for high stability impact
        stability_impact = abs(goal.get('stability_impact', 0))
        if stability_impact > 0.05:
            return min(0.85, base_threshold + 0.15)
        
        # Increase threshold for performance vs interpretability tradeoffs
        if goal.get('goal_type') == 'chess_performance' and stability_impact > 0.03:
            return min(0.80, base_threshold + 0.10)
        
        return base_threshold
    
    def _detect_conflicts(
        self,
        agent_opinions: List[AgentOpinion],
        vote_counts: Dict[str, int],
        variance: float
    ) -> List[str]:
        """Detect ethical conflicts in agent opinions"""
        conflicts = []
        
        # High disagreement
        if variance > 0.25:
            conflicts.append(f"High alignment variance detected (σ={variance:.2f})")
        
        # Split decision
        if vote_counts["approve"] > 0 and vote_counts["reject"] > 0:
            ratio = vote_counts["reject"] / (vote_counts["approve"] + vote_counts["reject"])
            if ratio >= 0.3:  # 30% dissent
                conflicts.append(f"Significant dissent: {vote_counts['reject']} reject vs {vote_counts['approve']} approve")
        
        # Low confidence
        low_confidence_count = sum(1 for op in agent_opinions if op.confidence < 0.6)
        if low_confidence_count >= len(agent_opinions) / 2:
            conflicts.append(f"{low_confidence_count} agents have low confidence (<0.6)")
        
        # Provider disagreement
        provider_votes = {}
        for op in agent_opinions:
            if op.provider not in provider_votes:
                provider_votes[op.provider] = []
            provider_votes[op.provider].append(op.vote)
        
        # Check if different providers strongly disagree
        if len(provider_votes) >= 2:
            provider_approvals = {
                provider: sum(1 for v in votes if v == "approve") / len(votes)
                for provider, votes in provider_votes.items()
            }
            
            if max(provider_approvals.values()) - min(provider_approvals.values()) > 0.4:
                conflicts.append("Strong provider disagreement detected")
        
        return conflicts
    
    def _generate_reasoning_summary(
        self,
        agent_opinions: List[AgentOpinion],
        vote_counts: Dict[str, int],
        agreement_score: float,
        consensus_reached: bool
    ) -> str:
        """Generate human-readable reasoning summary"""
        
        summary_parts = []
        
        # Consensus status
        if consensus_reached:
            summary_parts.append(f"✅ Consensus reached with {vote_counts['approve']} approvals, {vote_counts['reject']} rejections.")
        else:
            summary_parts.append(f"⚠️ No clear consensus: {vote_counts['approve']} approve, {vote_counts['reject']} reject, {vote_counts['abstain']} abstain.")
        
        # Alignment
        summary_parts.append(f"Ethical Alignment Index: {agreement_score:.2f}/1.00.")
        
        # Key reasoning points
        reasoning_snippets = [op.opinion[:100] for op in agent_opinions if op.vote == "approve"][:2]
        if reasoning_snippets:
            summary_parts.append(f"Supporting views: {' | '.join(reasoning_snippets)}")
        
        rejection_snippets = [op.opinion[:100] for op in agent_opinions if op.vote == "reject"][:1]
        if rejection_snippets:
            summary_parts.append(f"Dissenting views: {' | '.join(rejection_snippets)}")
        
        return " ".join(summary_parts)
    
    def _create_empty_consensus(
        self,
        consensus_id: str,
        timestamp: str,
        goal: Dict[str, Any]
    ) -> ConsensusResult:
        """Create empty consensus result for error cases"""
        return ConsensusResult(
            consensus_id=consensus_id,
            timestamp=timestamp,
            goal_id=goal.get('goal_id', 'unknown'),
            goal_description=goal.get('description', ''),
            total_agents=len(self.agent_configs),
            agents_participated=0,
            agent_opinions=[],
            vote_distribution={"approve": 0, "reject": 0, "abstain": 0},
            consensus_reached=False,
            consensus_threshold=self.base_threshold,
            agreement_score=0.0,
            agreement_variance=0.0,
            final_decision="requires_review",
            reasoning_summary="Error: Unable to gather agent opinions",
            conflicts_detected=["No agent responses available"],
            conflict_resolution=None
        )
    
    async def resolve_conflicts(
        self,
        consensus_result: ConsensusResult
    ) -> Tuple[ConsensusResult, str]:
        """
        Resolve ethical conflicts detected in consensus.
        
        Args:
            consensus_result: Initial consensus with conflicts
        
        Returns:
            Updated consensus result and resolution explanation
        """
        try:
            if not consensus_result.conflicts_detected:
                return consensus_result, "No conflicts to resolve"
            
            logger.info(f"Resolving {len(consensus_result.conflicts_detected)} conflicts...")
            
            resolution_strategies = []
            
            # Strategy 1: High variance → Request additional human review
            if any("variance" in c.lower() for c in consensus_result.conflicts_detected):
                resolution_strategies.append("High variance detected: Flagging for human review")
                consensus_result.final_decision = "requires_review"
            
            # Strategy 2: Provider disagreement → Weight by specialty
            if any("provider disagreement" in c.lower() for c in consensus_result.conflicts_detected):
                resolution_strategies.append("Provider disagreement: Using weighted vote by agent specialty")
                # Already handled by weighted voting
            
            # Strategy 3: Low confidence → Conservative approach
            if any("low confidence" in c.lower() for c in consensus_result.conflicts_detected):
                resolution_strategies.append("Low confidence: Applying conservative threshold")
                if consensus_result.agreement_score < 0.80:
                    consensus_result.final_decision = "requires_review"
            
            # Strategy 4: Significant dissent → Require manual approval
            if any("dissent" in c.lower() for c in consensus_result.conflicts_detected):
                resolution_strategies.append("Significant dissent: Requiring manual approval")
                if consensus_result.final_decision == "approved":
                    consensus_result.final_decision = "requires_review"
            
            conflict_resolution = " | ".join(resolution_strategies) if resolution_strategies else "Standard consensus protocol applied"
            consensus_result.conflict_resolution = conflict_resolution
            
            logger.info(f"Conflict resolution: {conflict_resolution}")
            return consensus_result, conflict_resolution
            
        except Exception as e:
            logger.error(f"Error resolving conflicts: {e}")
            return consensus_result, f"Error in conflict resolution: {str(e)}"
    
    async def refine_governance_rules(
        self,
        consensus_result: ConsensusResult,
        current_rules: List[Dict[str, Any]]
    ) -> List[RuleRefinement]:
        """
        Refine governance rules based on consensus outcomes.
        
        Adaptive refinement based on:
        - Consensus trends over time
        - Violation patterns
        - Agreement variance
        
        Args:
            consensus_result: Recent consensus result
            current_rules: Active governance rules
        
        Returns:
            List of rule refinements
        """
        try:
            refinements = []
            
            # Get recent consensus history
            recent_consensuses = await self.db.llm_ethics_consensus_log.find().sort(
                "timestamp", -1
            ).limit(10).to_list(10)
            
            if len(recent_consensuses) < 3:
                logger.info("Insufficient consensus history for rule refinement")
                return refinements
            
            # Calculate trends
            avg_alignment = np.mean([c.get('agreement_score', 0) for c in recent_consensuses])
            avg_variance = np.mean([c.get('agreement_variance', 0) for c in recent_consensuses])
            
            # Refine transparency rule
            transparency_rule = next((r for r in current_rules if "transparency" in r['name'].lower()), None)
            if transparency_rule:
                refinement = self._refine_transparency_rule(
                    transparency_rule, avg_alignment, avg_variance, consensus_result
                )
                if refinement:
                    refinements.append(refinement)
            
            # Refine fairness rule
            fairness_rule = next((r for r in current_rules if "fairness" in r['name'].lower()), None)
            if fairness_rule:
                refinement = self._refine_fairness_rule(
                    fairness_rule, avg_alignment, recent_consensuses, consensus_result
                )
                if refinement:
                    refinements.append(refinement)
            
            # Refine safety rules
            safety_rules = [r for r in current_rules if r['constraint_type'] == 'safety']
            for rule in safety_rules:
                refinement = self._refine_safety_rule(
                    rule, avg_variance, recent_consensuses, consensus_result
                )
                if refinement:
                    refinements.append(refinement)
            
            logger.info(f"Generated {len(refinements)} rule refinements")
            return refinements
            
        except Exception as e:
            logger.error(f"Error refining governance rules: {e}")
            return []
    
    def _refine_transparency_rule(
        self,
        rule: Dict[str, Any],
        avg_alignment: float,
        avg_variance: float,
        consensus: ConsensusResult
    ) -> Optional[RuleRefinement]:
        """Refine transparency rule based on consensus patterns"""
        
        # If low alignment, strengthen transparency requirements
        current_weight = rule.get('threshold', 0.0)
        
        if avg_alignment < 0.75 and avg_variance > 0.20:
            # Strengthen transparency (lower threshold = stricter)
            new_weight = max(0.0, current_weight - 0.05)
            delta = new_weight - current_weight
            
            if abs(delta) > 0.01:  # Only if meaningful change
                return RuleRefinement(
                    refinement_id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    rule_id=rule['rule_id'],
                    rule_name=rule['name'],
                    previous_weight=current_weight,
                    new_weight=new_weight,
                    weight_delta=delta,
                    refinement_reason=f"Low alignment ({avg_alignment:.2f}) and high variance ({avg_variance:.2f}) indicate need for stronger transparency",
                    consensus_id=consensus.consensus_id,
                    auto_applied=False,
                    requires_approval=True
                )
        
        return None
    
    def _refine_fairness_rule(
        self,
        rule: Dict[str, Any],
        avg_alignment: float,
        recent_consensuses: List[Dict[str, Any]],
        consensus: ConsensusResult
    ) -> Optional[RuleRefinement]:
        """Refine fairness rule based on conflict patterns"""
        
        # Check if fairness-related conflicts are frequent
        fairness_conflicts = sum(
            1 for c in recent_consensuses
            if any("fairness" in conf.lower() or "bias" in conf.lower() 
                   for conf in c.get('conflicts_detected', []))
        )
        
        current_weight = rule.get('threshold', 0.15)
        
        if fairness_conflicts >= 3:  # Frequent fairness concerns
            # Tighten fairness threshold
            new_weight = max(0.05, current_weight - 0.03)
            delta = new_weight - current_weight
            
            if abs(delta) > 0.01:
                return RuleRefinement(
                    refinement_id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    rule_id=rule['rule_id'],
                    rule_name=rule['name'],
                    previous_weight=current_weight,
                    new_weight=new_weight,
                    weight_delta=delta,
                    refinement_reason=f"Frequent fairness conflicts detected ({fairness_conflicts} in last 10 consensuses)",
                    consensus_id=consensus.consensus_id,
                    auto_applied=False,
                    requires_approval=True
                )
        
        return None
    
    def _refine_safety_rule(
        self,
        rule: Dict[str, Any],
        avg_variance: float,
        recent_consensuses: List[Dict[str, Any]],
        consensus: ConsensusResult
    ) -> Optional[RuleRefinement]:
        """Refine safety rule based on stability metrics"""
        
        # Check if stability violations are increasing
        stability_violations = sum(
            1 for c in recent_consensuses
            if any("stability" in conf.lower() for conf in c.get('conflicts_detected', []))
        )
        
        current_weight = rule.get('threshold', 0.05)
        
        if stability_violations >= 2 or avg_variance > 0.25:
            # Tighten stability bounds
            new_weight = max(0.03, current_weight - 0.01)
            delta = new_weight - current_weight
            
            if abs(delta) > 0.005:
                return RuleRefinement(
                    refinement_id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    rule_id=rule['rule_id'],
                    rule_name=rule['name'],
                    previous_weight=current_weight,
                    new_weight=new_weight,
                    weight_delta=delta,
                    refinement_reason=f"Stability concerns: {stability_violations} violations, variance {avg_variance:.2f}",
                    consensus_id=consensus.consensus_id,
                    auto_applied=False,
                    requires_approval=True
                )
        
        return None
    
    async def generate_ethics_report(
        self,
        consensus_id: Optional[str] = None,
        lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Generate comprehensive ethics report summarizing consensus outcomes and rule shifts.
        
        Args:
            consensus_id: Specific consensus to report on (optional)
            lookback_hours: Hours of history to analyze
        
        Returns:
            Comprehensive ethics report
        """
        try:
            report_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Get consensus logs
            if consensus_id:
                consensuses = await self.db.llm_ethics_consensus_log.find(
                    {"consensus_id": consensus_id}
                ).to_list(1)
            else:
                cutoff_time = datetime.now(timezone.utc).timestamp() - (lookback_hours * 3600)
                consensuses = await self.db.llm_ethics_consensus_log.find().sort(
                    "timestamp", -1
                ).to_list(100)
            
            # Get rule refinements
            refinements = await self.db.llm_rule_refinement.find().sort(
                "timestamp", -1
            ).limit(20).to_list(20)
            
            # Calculate aggregate metrics
            if consensuses:
                total_consensuses = len(consensuses)
                avg_eai = np.mean([c.get('agreement_score', 0) for c in consensuses])
                avg_variance = np.mean([c.get('agreement_variance', 0) for c in consensuses])
                
                # Decision distribution
                decision_dist = {}
                for c in consensuses:
                    decision = c.get('final_decision', 'unknown')
                    decision_dist[decision] = decision_dist.get(decision, 0) + 1
                
                # Conflict analysis
                total_conflicts = sum(len(c.get('conflicts_detected', [])) for c in consensuses)
                
                # Agent participation
                agent_participation = {}
                for c in consensuses:
                    agents = c.get('agents_participated', 0)
                    agent_participation[agents] = agent_participation.get(agents, 0) + 1
                
                # Provider analysis
                provider_votes = {"openai": [], "anthropic": [], "gemini": []}
                for c in consensuses:
                    for opinion in c.get('agent_opinions', []):
                        provider = opinion.get('provider')
                        if provider in provider_votes:
                            provider_votes[provider].append(opinion.get('alignment_score', 0))
                
                provider_avg_alignment = {
                    provider: np.mean(scores) if scores else 0.0
                    for provider, scores in provider_votes.items()
                }
            else:
                total_consensuses = 0
                avg_eai = 0.0
                avg_variance = 0.0
                decision_dist = {}
                total_conflicts = 0
                agent_participation = {}
                provider_avg_alignment = {}
            
            # Rule refinement summary
            refinement_summary = {
                "total_refinements": len(refinements),
                "pending_approval": sum(1 for r in refinements if r.get('requires_approval', True) and not r.get('auto_applied', False)),
                "auto_applied": sum(1 for r in refinements if r.get('auto_applied', False)),
                "recent_changes": [
                    {
                        "rule_name": r.get('rule_name'),
                        "weight_delta": r.get('weight_delta'),
                        "reason": r.get('refinement_reason')[:100]
                    }
                    for r in refinements[:5]
                ]
            }
            
            # System health assessment
            health_status = self._assess_consensus_health(avg_eai, avg_variance, total_conflicts, total_consensuses)
            
            report = {
                "report_id": report_id,
                "generated_at": timestamp,
                "lookback_hours": lookback_hours,
                "consensus_summary": {
                    "total_consensuses": total_consensuses,
                    "avg_ethical_alignment_index": round(avg_eai, 3),
                    "avg_agreement_variance": round(avg_variance, 3),
                    "decision_distribution": decision_dist,
                    "total_conflicts": total_conflicts,
                    "agent_participation": agent_participation
                },
                "provider_analysis": {
                    "avg_alignment_by_provider": provider_avg_alignment,
                    "provider_diversity": len([p for p, scores in provider_votes.items() if scores])
                },
                "rule_refinement": refinement_summary,
                "system_health": health_status,
                "recommendations": self._generate_recommendations(
                    avg_eai, avg_variance, total_conflicts, total_consensuses, refinements
                )
            }
            
            logger.info(f"Generated ethics report: EAI={avg_eai:.2f}, health={health_status['status']}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating ethics report: {e}")
            return {
                "report_id": str(uuid.uuid4()),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }
    
    def _assess_consensus_health(
        self,
        avg_eai: float,
        avg_variance: float,
        total_conflicts: int,
        total_consensuses: int
    ) -> Dict[str, Any]:
        """Assess overall consensus system health"""
        
        # Health score calculation
        eai_score = avg_eai * 100  # 0-100
        variance_penalty = avg_variance * 50  # Lower is better
        conflict_rate = (total_conflicts / total_consensuses) if total_consensuses > 0 else 0
        conflict_penalty = conflict_rate * 30
        
        health_score = max(0, min(100, eai_score - variance_penalty - conflict_penalty))
        
        if health_score >= 85:
            status = "excellent"
            description = "Consensus system operating optimally with high alignment and low conflict"
        elif health_score >= 70:
            status = "good"
            description = "Consensus system functioning well with acceptable variance"
        elif health_score >= 55:
            status = "needs_attention"
            description = "Elevated variance or conflicts detected, monitoring recommended"
        else:
            status = "critical"
            description = "Significant consensus challenges, intervention required"
        
        return {
            "health_score": round(health_score, 1),
            "status": status,
            "description": description,
            "avg_eai": round(avg_eai, 3),
            "avg_variance": round(avg_variance, 3),
            "conflict_rate": round(conflict_rate, 3)
        }
    
    def _generate_recommendations(
        self,
        avg_eai: float,
        avg_variance: float,
        total_conflicts: int,
        total_consensuses: int,
        refinements: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if avg_eai < 0.75:
            recommendations.append("⚠️ Low average EAI detected. Consider reviewing goal formulation criteria.")
        
        if avg_variance > 0.25:
            recommendations.append("⚠️ High agreement variance. Agents show significant disagreement - review governance rules.")
        
        conflict_rate = (total_conflicts / total_consensuses) if total_consensuses > 0 else 0
        if conflict_rate > 0.5:
            recommendations.append("⚠️ High conflict rate (>50%). Strengthen ethical guidelines or adjust consensus threshold.")
        
        pending_refinements = sum(1 for r in refinements if r.get('requires_approval', True) and not r.get('auto_applied', False))
        if pending_refinements > 5:
            recommendations.append(f"📋 {pending_refinements} rule refinements pending approval. Review and apply to improve alignment.")
        
        if not recommendations:
            recommendations.append("✅ Consensus system operating within optimal parameters. Continue monitoring.")
        
        return recommendations
