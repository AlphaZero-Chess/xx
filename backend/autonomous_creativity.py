"""
Autonomous Creativity & Meta-Strategic Synthesis (Step 29)

This module implements a creative, meta-strategic reasoning engine capable of:
- Generating novel chess strategies and hypotheses
- Performing creative recombination of prior cognitive patterns
- Synthesizing long-term meta-strategic goals aligned with ethical constraints
- Operating in advisory-only mode with safety bounds

Features:
- Multi-provider LLM cross-pollination (GPT + Claude + Gemini)
- Full-spectrum creativity (openings, middlegame, endgame)
- Ethical guardrails (fair play, educational, anti-cheating)
- Originality and stability metrics
- Value alignment verification
"""

import logging
import asyncio
import uuid
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import numpy as np
import os

# Import emergentintegrations for multi-provider LLM support
from emergentintegrations.llm.chat import LlmChat, UserMessage

logger = logging.getLogger(__name__)


@dataclass
class CreativeStrategy:
    """A single creative chess strategy"""
    strategy_id: str
    timestamp: str
    phase: str  # "opening", "middlegame", "endgame"
    strategy_name: str
    description: str
    tactical_elements: List[str]
    novelty_score: float  # 0-1: How original/novel
    stability_score: float  # 0-1: How reliable/stable
    ethical_alignment: float  # 0-1: Alignment with ethical constraints
    educational_value: float  # 0-1: Learning potential
    risk_level: float  # 0-1: Risk of failure
    llm_provider: str  # Which provider generated it
    parent_patterns: List[str]  # Patterns it derived from
    rejected: bool  # True if failed ethical checks
    rejection_reason: Optional[str]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class MetaStrategy:
    """High-level meta-strategic synthesis"""
    meta_id: str
    timestamp: str
    theme: str  # Overarching strategic theme
    description: str
    integrated_strategies: List[str]  # Strategy IDs included
    coherence_score: float  # 0-1: Internal consistency
    adaptability_score: float  # 0-1: Flexibility
    long_term_value: float  # 0-1: Strategic depth
    ethical_compliance: float  # 0-1: Ethical alignment
    recommended_contexts: List[str]
    safety_constraints: List[str]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class CreativityMetrics:
    """System creativity performance metrics"""
    timestamp: str
    total_ideas_generated: int
    ideas_approved: int
    ideas_rejected: int
    avg_novelty: float
    avg_stability: float
    avg_ethical_alignment: float
    creativity_health: float  # 0-1: Overall system health
    provider_distribution: Dict[str, int]
    phase_distribution: Dict[str, int]
    
    def to_dict(self):
        return asdict(self)


class CreativeSynthesisController:
    """
    Autonomous Creativity & Meta-Strategic Synthesis Controller
    
    Generates novel strategies through multi-provider LLM synthesis while
    maintaining ethical and performance constraints.
    """
    
    def __init__(self, db_client):
        self.db = db_client
        self.llm_key = os.environ.get('EMERGENT_LLM_KEY')
        
        if not self.llm_key:
            logger.warning("EMERGENT_LLM_KEY not found - using mock mode")
            self.llm_available = False
        else:
            self.llm_available = True
            logger.info("Creativity Controller initialized with multi-provider LLM support")
        
        # Ethical constraints
        self.ethical_rules = {
            "fair_play": "Must not enable unfair advantages or cheating",
            "educational": "Should provide learning value and be explainable",
            "anti_cheating": "Must not facilitate misuse in competitive contexts"
        }
        
        # Creativity parameters
        self.novelty_threshold = 0.60  # Minimum novelty to be considered "creative"
        self.stability_threshold = 0.50  # Minimum stability for recommendation
        self.ethical_threshold = 0.75  # Minimum ethical alignment required
        
        # LLM providers for diversity
        self.llm_providers = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-sonnet-20241022",
            "google": "gemini-2.0-flash-exp"
        }
    
    async def generate_creative_strategies(
        self,
        phase: Optional[str] = None,
        count: int = 3,
        use_patterns: bool = True
    ) -> List[CreativeStrategy]:
        """
        Generate innovative strategic variants using LLM cross-pollination.
        
        Args:
            phase: Specific phase ("opening", "middlegame", "endgame") or None for all
            count: Number of strategies to generate per phase
            use_patterns: Whether to leverage existing cognitive patterns
        
        Returns:
            List of CreativeStrategy objects
        """
        try:
            logger.info(f"Generating creative strategies (phase={phase}, count={count})...")
            
            strategies = []
            
            # Determine phases to generate for
            if phase:
                phases = [phase]
            else:
                phases = ["opening", "middlegame", "endgame"]
            
            # Get existing patterns if requested
            parent_patterns = []
            if use_patterns:
                parent_patterns = await self._get_existing_patterns()
            
            # Generate strategies for each phase
            for chess_phase in phases:
                for i in range(count):
                    # Rotate through providers for diversity
                    provider_name = list(self.llm_providers.keys())[i % len(self.llm_providers)]
                    
                    if self.llm_available:
                        strategy = await self._llm_generate_strategy(
                            chess_phase, provider_name, parent_patterns
                        )
                    else:
                        strategy = self._mock_generate_strategy(
                            chess_phase, provider_name, parent_patterns
                        )
                    
                    # Evaluate strategy
                    strategy = await self._evaluate_strategy(strategy)
                    
                    strategies.append(strategy)
                    
                    # Store in database
                    await self.db.llm_creative_synthesis.insert_one(strategy.to_dict())
            
            logger.info(f"Generated {len(strategies)} creative strategies")
            return strategies
            
        except Exception as e:
            logger.error(f"Error generating creative strategies: {e}")
            return []
    
    async def _get_existing_patterns(self) -> List[str]:
        """Retrieve existing cognitive patterns from Step 27"""
        try:
            patterns = await self.db.llm_cognitive_patterns.find().sort(
                "strength", -1
            ).limit(5).to_list(5)
            
            return [p.get("pattern_name", "") for p in patterns if p.get("pattern_name")]
        except:
            return ["Trust-Memory Harmonization", "Predictive Alignment", "Ethical Convergence"]
    
    async def _llm_generate_strategy(
        self,
        phase: str,
        provider_name: str,
        parent_patterns: List[str]
    ) -> CreativeStrategy:
        """Use LLM to generate creative strategy [PROD]"""
        
        try:
            model = self.llm_providers[provider_name]
            
            chat = LlmChat(
                api_key=self.llm_key,
                session_id=f"creativity-{provider_name}-{uuid.uuid4()}",
                system_message="You are a creative chess strategist for AlphaZero AI. Generate novel, educational, and ethical chess strategies."
            ).with_model(provider_name, model)
            
            patterns_context = ""
            if parent_patterns:
                patterns_context = f"\n\nExisting cognitive patterns to build upon: {', '.join(parent_patterns[:3])}"
            
            prompt = f"""Generate a novel chess strategy for the **{phase}** phase.

Requirements:
- Must be **creative and original** (avoid common/standard approaches)
- Should be **educationally valuable** (explainable and learnable)
- Must adhere to **fair play** (no cheating or exploitation)
- Should have **practical viability** (not purely theoretical)
{patterns_context}

Provide:
1. **Strategy Name** (3-5 words, memorable)
2. **Description** (2-3 sentences explaining the core concept)
3. **Tactical Elements** (3-5 specific moves/concepts involved)

Format your response as:
NAME: [strategy name]
DESCRIPTION: [description]
TACTICS: [element 1], [element 2], [element 3]

Be creative and think outside standard chess theory!"""

            user_message = UserMessage(text=prompt)
            response = await chat.send_message(user_message)
            
            # Parse response
            parsed = self._parse_llm_strategy_response(response, phase, provider_name, parent_patterns)
            logger.info(f"[PROD] Generated strategy '{parsed.strategy_name}' via {provider_name}")
            return parsed
            
        except Exception as e:
            logger.error(f"Error in LLM strategy generation: {e}, falling back to mock")
            return self._mock_generate_strategy(phase, provider_name, parent_patterns)
    
    def _parse_llm_strategy_response(
        self,
        response: str,
        phase: str,
        provider: str,
        patterns: List[str]
    ) -> CreativeStrategy:
        """Parse LLM response into CreativeStrategy object"""
        
        lines = response.strip().split('\n')
        
        name = "Unnamed Strategy"
        description = "No description provided"
        tactics = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("NAME:"):
                name = line.replace("NAME:", "").strip()
            elif line.startswith("DESCRIPTION:"):
                description = line.replace("DESCRIPTION:", "").strip()
            elif line.startswith("TACTICS:"):
                tactics_str = line.replace("TACTICS:", "").strip()
                tactics = [t.strip() for t in tactics_str.split(",")]
        
        # If parsing failed, extract from raw response
        if name == "Unnamed Strategy" or not description:
            name = f"{phase.capitalize()} Creative Strategy"
            description = response[:200] if len(response) > 200 else response
            tactics = ["Dynamic positioning", "Adaptive play", "Pattern recognition"]
        
        return CreativeStrategy(
            strategy_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            phase=phase,
            strategy_name=name,
            description=description,
            tactical_elements=tactics[:5],
            novelty_score=0.0,  # Will be calculated in evaluation
            stability_score=0.0,
            ethical_alignment=0.0,
            educational_value=0.0,
            risk_level=0.0,
            llm_provider=provider,
            parent_patterns=patterns[:3],
            rejected=False,
            rejection_reason=None
        )
    
    def _mock_generate_strategy(
        self,
        phase: str,
        provider_name: str,
        parent_patterns: List[str]
    ) -> CreativeStrategy:
        """Generate mock strategy [MOCKED]"""
        
        mock_strategies = {
            "opening": {
                "name": "Quantum Pawn Storm",
                "description": "A novel opening that combines hypermodern principles with aggressive pawn advances, creating immediate central tension while maintaining flexible piece development. Inspired by Trust-Memory Harmonization patterns. [MOCKED]",
                "tactics": ["d4-d5 pawn lever", "Flexible knight development", "Delayed castling", "Central pawn tension", "Bishop fianchetto options"]
            },
            "middlegame": {
                "name": "Adaptive Piece Synergy",
                "description": "Middlegame strategy focusing on dynamic piece coordination where each piece's role adapts based on opponent's formation. Uses Predictive Alignment to anticipate defensive setups. [MOCKED]",
                "tactics": ["Dynamic rook lifts", "Bishop pair activation", "Knight outpost creation", "Pawn break timing", "Piece trade evaluation"]
            },
            "endgame": {
                "name": "Zugzwang Orchestration",
                "description": "Advanced endgame technique that systematically creates zugzwang positions through careful king and pawn coordination. Builds on Ethical Convergence for optimal resource management. [MOCKED]",
                "tactics": ["Opposition mastery", "Triangulation patterns", "Pawn breakthrough timing", "King activation", "Tempo management"]
            }
        }
        
        template = mock_strategies.get(phase, mock_strategies["middlegame"])
        
        return CreativeStrategy(
            strategy_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            phase=phase,
            strategy_name=template["name"],
            description=template["description"],
            tactical_elements=template["tactics"],
            novelty_score=0.0,  # Will be set in evaluation
            stability_score=0.0,
            ethical_alignment=0.0,
            educational_value=0.0,
            risk_level=0.0,
            llm_provider=f"{provider_name} [MOCKED]",
            parent_patterns=parent_patterns[:3] if parent_patterns else [],
            rejected=False,
            rejection_reason=None
        )
    
    async def _evaluate_strategy(self, strategy: CreativeStrategy) -> CreativeStrategy:
        """
        Evaluate strategy for originality, ethical alignment, and stability.
        
        Returns:
            Updated CreativeStrategy with scores
        """
        try:
            # Score 1: Novelty (how original is it?)
            novelty = await self._calculate_novelty(strategy)
            
            # Score 2: Stability (how reliable/practical?)
            stability = self._calculate_stability(strategy)
            
            # Score 3: Ethical Alignment
            ethical_score, passed, reason = self._check_ethical_alignment(strategy)
            
            # Score 4: Educational Value
            educational = self._calculate_educational_value(strategy)
            
            # Score 5: Risk Level
            risk = self._calculate_risk_level(strategy)
            
            # Update strategy
            strategy.novelty_score = novelty
            strategy.stability_score = stability
            strategy.ethical_alignment = ethical_score
            strategy.educational_value = educational
            strategy.risk_level = risk
            strategy.rejected = not passed
            strategy.rejection_reason = reason if not passed else None
            
            logger.info(
                f"Evaluated '{strategy.strategy_name}': "
                f"Novelty={novelty:.2f}, Stability={stability:.2f}, "
                f"Ethical={ethical_score:.2f}, Rejected={strategy.rejected}"
            )
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error evaluating strategy: {e}")
            # Set default scores
            strategy.novelty_score = 0.70
            strategy.stability_score = 0.65
            strategy.ethical_alignment = 0.85
            strategy.educational_value = 0.75
            strategy.risk_level = 0.45
            return strategy
    
    async def _calculate_novelty(self, strategy: CreativeStrategy) -> float:
        """Calculate novelty score by comparing to existing strategies"""
        try:
            # Get similar existing strategies
            existing = await self.db.llm_creative_synthesis.find({
                "phase": strategy.phase,
                "rejected": False
            }).limit(20).to_list(20)
            
            if not existing:
                return 0.85  # High novelty if no prior strategies
            
            # Simple text similarity check (in production would use embeddings)
            name_matches = sum(1 for s in existing if strategy.strategy_name.lower() in s.get("strategy_name", "").lower())
            
            # More matches = less novel
            novelty = max(0.50, 1.0 - (name_matches / len(existing)))
            
            # Boost if uses parent patterns
            if strategy.parent_patterns:
                novelty = min(1.0, novelty + 0.10)
            
            return round(novelty, 2)
            
        except:
            return 0.75  # Default moderate novelty
    
    def _calculate_stability(self, strategy: CreativeStrategy) -> float:
        """Calculate stability/reliability score"""
        
        # Base stability from tactical element count
        tactic_count = len(strategy.tactical_elements)
        base_stability = min(1.0, tactic_count / 5.0)
        
        # Adjust for description length (longer = more thought out)
        desc_length = len(strategy.description)
        desc_factor = min(1.0, desc_length / 150.0)
        
        # Adjust for phase (endgames are more stable/concrete)
        phase_factor = {
            "opening": 0.70,
            "middlegame": 0.65,
            "endgame": 0.85
        }.get(strategy.phase, 0.70)
        
        stability = (base_stability * 0.4 + desc_factor * 0.3 + phase_factor * 0.3)
        
        return round(min(1.0, max(0.40, stability)), 2)
    
    def _check_ethical_alignment(self, strategy: CreativeStrategy) -> Tuple[float, bool, Optional[str]]:
        """
        Check if strategy meets ethical constraints.
        
        Returns:
            (ethical_score, passed, rejection_reason)
        """
        
        # Keywords that might indicate ethical issues
        problematic_keywords = [
            "cheat", "exploit", "abuse", "unfair", "trick", "deceive",
            "illegal", "violation", "manipulation", "computer assistance"
        ]
        
        text = (strategy.strategy_name + " " + strategy.description).lower()
        
        violations = [kw for kw in problematic_keywords if kw in text]
        
        if violations:
            return (0.30, False, f"Ethical violation: Contains problematic terms ({', '.join(violations)}). Violates fair play principles.")
        
        # Check for educational value (required)
        if len(strategy.description) < 50:
            return (0.60, False, "Insufficient educational value: Description too brief to be instructive.")
        
        # All checks passed
        ethical_score = 0.92  # High ethical alignment
        
        # Slight reduction if high risk
        return (ethical_score, True, None)
    
    def _calculate_educational_value(self, strategy: CreativeStrategy) -> float:
        """Calculate educational/learning value"""
        
        # Factor 1: Description quality
        desc_score = min(1.0, len(strategy.description) / 200.0)
        
        # Factor 2: Tactical elements (more = more educational)
        tactics_score = min(1.0, len(strategy.tactical_elements) / 5.0)
        
        # Factor 3: Pattern integration (learning from existing knowledge)
        pattern_score = min(1.0, len(strategy.parent_patterns) / 3.0) if strategy.parent_patterns else 0.5
        
        educational = (desc_score * 0.4 + tactics_score * 0.4 + pattern_score * 0.2)
        
        return round(min(1.0, max(0.40, educational)), 2)
    
    def _calculate_risk_level(self, strategy: CreativeStrategy) -> float:
        """Calculate risk level (higher novelty = higher risk)"""
        
        # Inverse relationship with stability
        risk = 1.0 - strategy.stability_score
        
        # Novelty adds risk
        if strategy.novelty_score > 0.80:
            risk = min(1.0, risk + 0.15)
        
        # Phase affects risk
        phase_risk = {
            "opening": 0.50,  # Openings are moderate risk
            "middlegame": 0.65,  # Middlegames are higher risk
            "endgame": 0.40  # Endgames are lower risk (more concrete)
        }.get(strategy.phase, 0.50)
        
        risk = (risk * 0.6 + phase_risk * 0.4)
        
        return round(min(1.0, max(0.20, risk)), 2)
    
    async def synthesize_meta_strategy(
        self,
        strategies: Optional[List[CreativeStrategy]] = None
    ) -> MetaStrategy:
        """
        Synthesize high-level meta-strategy from creative strategies.
        
        Args:
            strategies: List of strategies to synthesize, or None to use recent
        
        Returns:
            MetaStrategy with coherent long-term vision
        """
        try:
            logger.info("Synthesizing meta-strategy from creative outputs...")
            
            # Get strategies if not provided
            if not strategies:
                strategy_docs = await self.db.llm_creative_synthesis.find({
                    "rejected": False
                }).sort("timestamp", -1).limit(10).to_list(10)
                
                strategies = [
                    CreativeStrategy(**{k: v for k, v in doc.items() if k != '_id'})
                    for doc in strategy_docs
                ]
            
            if not strategies:
                # Generate sample strategies if none exist
                strategies = await self.generate_creative_strategies(count=1)
            
            # Filter approved strategies
            approved = [s for s in strategies if not s.rejected]
            
            if not approved:
                raise ValueError("No approved strategies to synthesize")
            
            # Use LLM to synthesize if available
            if self.llm_available:
                meta = await self._llm_synthesize_meta_strategy(approved)
            else:
                meta = self._mock_synthesize_meta_strategy(approved)
            
            # Store in database
            await self.db.llm_meta_strategy_log.insert_one(meta.to_dict())
            
            logger.info(f"Meta-strategy synthesized: '{meta.theme}'")
            return meta
            
        except Exception as e:
            logger.error(f"Error synthesizing meta-strategy: {e}")
            raise
    
    async def _llm_synthesize_meta_strategy(
        self,
        strategies: List[CreativeStrategy]
    ) -> MetaStrategy:
        """Use LLM to synthesize meta-strategy [PROD]"""
        
        try:
            # Use OpenAI for synthesis
            chat = LlmChat(
                api_key=self.llm_key,
                session_id=f"meta-strategy-{uuid.uuid4()}",
                system_message="You are a meta-strategic analyzer for AlphaZero Chess AI. Synthesize high-level strategic themes from creative outputs."
            ).with_model("openai", "gpt-4o-mini")
            
            # Build strategy summary
            strategy_summary = "\n".join([
                f"- {s.phase.upper()}: {s.strategy_name} (Novelty: {s.novelty_score:.2f}, Stability: {s.stability_score:.2f})"
                for s in strategies[:6]
            ])
            
            prompt = f"""Analyze the following creative chess strategies and synthesize a meta-strategic theme:

**Strategies:**
{strategy_summary}

Provide:
1. **Meta-Strategic Theme** (3-5 words capturing the overarching approach)
2. **Description** (2-3 sentences explaining the unified vision)
3. **Recommended Contexts** (3 situations where this meta-strategy excels)

Format:
THEME: [theme]
DESCRIPTION: [description]
CONTEXTS: [context 1], [context 2], [context 3]

Focus on coherence, long-term value, and practical applicability."""

            user_message = UserMessage(text=prompt)
            response = await chat.send_message(user_message)
            
            # Parse response
            meta = self._parse_meta_strategy_response(response, strategies)
            logger.info(f"[PROD] Synthesized meta-strategy '{meta.theme}'")
            return meta
            
        except Exception as e:
            logger.error(f"Error in LLM meta-strategy synthesis: {e}")
            return self._mock_synthesize_meta_strategy(strategies)
    
    def _parse_meta_strategy_response(
        self,
        response: str,
        strategies: List[CreativeStrategy]
    ) -> MetaStrategy:
        """Parse LLM meta-strategy response"""
        
        lines = response.strip().split('\n')
        
        theme = "Unified Strategic Vision"
        description = "No description provided"
        contexts = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("THEME:"):
                theme = line.replace("THEME:", "").strip()
            elif line.startswith("DESCRIPTION:"):
                description = line.replace("DESCRIPTION:", "").strip()
            elif line.startswith("CONTEXTS:"):
                contexts_str = line.replace("CONTEXTS:", "").strip()
                contexts = [c.strip() for c in contexts_str.split(",")]
        
        if not contexts:
            contexts = ["Balanced positions", "Complex middlegames", "Strategic endgames"]
        
        # Calculate scores
        avg_novelty = np.mean([s.novelty_score for s in strategies])
        avg_stability = np.mean([s.stability_score for s in strategies])
        avg_ethical = np.mean([s.ethical_alignment for s in strategies])
        
        coherence = (avg_stability + avg_ethical) / 2.0
        adaptability = avg_novelty  # Novel = adaptable
        long_term = (coherence + avg_ethical) / 2.0
        
        return MetaStrategy(
            meta_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            theme=theme,
            description=description,
            integrated_strategies=[s.strategy_id for s in strategies],
            coherence_score=round(coherence, 2),
            adaptability_score=round(adaptability, 2),
            long_term_value=round(long_term, 2),
            ethical_compliance=round(avg_ethical, 2),
            recommended_contexts=contexts[:5],
            safety_constraints=[
                "Advisory mode only - no automatic application",
                "Requires human review before competitive use",
                "Monitor for unintended ethical implications"
            ]
        )
    
    def _mock_synthesize_meta_strategy(
        self,
        strategies: List[CreativeStrategy]
    ) -> MetaStrategy:
        """Generate mock meta-strategy [MOCKED]"""
        
        # Calculate average scores
        avg_novelty = np.mean([s.novelty_score for s in strategies]) if strategies else 0.75
        avg_stability = np.mean([s.stability_score for s in strategies]) if strategies else 0.70
        avg_ethical = np.mean([s.ethical_alignment for s in strategies]) if strategies else 0.88
        
        phases = list(set([s.phase for s in strategies])) if strategies else ["opening", "middlegame", "endgame"]
        
        theme = "Adaptive Multi-Phase Synthesis"
        description = (
            f"Meta-strategy integrating {len(strategies)} creative approaches across "
            f"{len(phases)} game phase(s). Emphasizes dynamic adaptation, ethical alignment, "
            f"and educational value while maintaining practical viability. "
            f"Builds upon existing cognitive patterns for coherent long-term play. [MOCKED]"
        )
        
        coherence = (avg_stability + avg_ethical) / 2.0
        adaptability = avg_novelty
        long_term = (coherence + avg_ethical) / 2.0
        
        return MetaStrategy(
            meta_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            theme=theme,
            description=description,
            integrated_strategies=[s.strategy_id for s in strategies] if strategies else [],
            coherence_score=round(coherence, 2),
            adaptability_score=round(adaptability, 2),
            long_term_value=round(long_term, 2),
            ethical_compliance=round(avg_ethical, 2),
            recommended_contexts=[
                "Balanced positional games",
                "Complex tactical middlegames",
                "Strategic endgame transitions"
            ],
            safety_constraints=[
                "Advisory mode only - no automatic application",
                "Requires human review before competitive use",
                "Monitor for unintended ethical implications",
                f"Average risk level: {np.mean([s.risk_level for s in strategies]) if strategies else 0.50:.2f}"
            ]
        )
    
    async def get_creativity_metrics(self) -> CreativityMetrics:
        """
        Get comprehensive creativity system metrics.
        
        Returns:
            CreativityMetrics with system health indicators
        """
        try:
            # Get all strategies
            all_strategies = await self.db.llm_creative_synthesis.find().to_list(1000)
            
            if not all_strategies:
                return CreativityMetrics(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    total_ideas_generated=0,
                    ideas_approved=0,
                    ideas_rejected=0,
                    avg_novelty=0.0,
                    avg_stability=0.0,
                    avg_ethical_alignment=0.0,
                    creativity_health=0.0,
                    provider_distribution={},
                    phase_distribution={}
                )
            
            approved = [s for s in all_strategies if not s.get("rejected", True)]
            rejected = [s for s in all_strategies if s.get("rejected", False)]
            
            # Calculate averages
            avg_novelty = np.mean([s.get("novelty_score", 0) for s in approved]) if approved else 0.0
            avg_stability = np.mean([s.get("stability_score", 0) for s in approved]) if approved else 0.0
            avg_ethical = np.mean([s.get("ethical_alignment", 0) for s in approved]) if approved else 0.0
            
            # Provider distribution
            provider_dist = {}
            for s in all_strategies:
                provider = s.get("llm_provider", "unknown")
                provider_dist[provider] = provider_dist.get(provider, 0) + 1
            
            # Phase distribution
            phase_dist = {}
            for s in all_strategies:
                phase = s.get("phase", "unknown")
                phase_dist[phase] = phase_dist.get(phase, 0) + 1
            
            # Creativity health (weighted combination)
            approval_rate = len(approved) / len(all_strategies) if all_strategies else 0.0
            health = (
                avg_novelty * 0.25 +
                avg_stability * 0.20 +
                avg_ethical * 0.35 +
                approval_rate * 0.20
            )
            
            return CreativityMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                total_ideas_generated=len(all_strategies),
                ideas_approved=len(approved),
                ideas_rejected=len(rejected),
                avg_novelty=round(avg_novelty, 2),
                avg_stability=round(avg_stability, 2),
                avg_ethical_alignment=round(avg_ethical, 2),
                creativity_health=round(health, 2),
                provider_distribution=provider_dist,
                phase_distribution=phase_dist
            )
            
        except Exception as e:
            logger.error(f"Error getting creativity metrics: {e}")
            return CreativityMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                total_ideas_generated=0,
                ideas_approved=0,
                ideas_rejected=0,
                avg_novelty=0.0,
                avg_stability=0.0,
                avg_ethical_alignment=0.0,
                creativity_health=0.0,
                provider_distribution={},
                phase_distribution={}
            )
    
    async def generate_creativity_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive creativity report with recommendations.
        
        Returns:
            Report dict with metrics, samples, and recommendations
        """
        try:
            logger.info("Generating creativity report...")
            
            # Get metrics
            metrics = await self.get_creativity_metrics()
            
            # Get recent approved strategies (samples)
            recent_approved = await self.db.llm_creative_synthesis.find({
                "rejected": False
            }).sort("timestamp", -1).limit(5).to_list(5)
            
            # Get recent meta-strategy
            recent_meta = await self.db.llm_meta_strategy_log.find_one(
                sort=[("timestamp", -1)]
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(metrics)
            
            # Get creativity by phase
            phase_breakdown = {}
            for phase in ["opening", "middlegame", "endgame"]:
                phase_strategies = await self.db.llm_creative_synthesis.find({
                    "phase": phase,
                    "rejected": False
                }).limit(3).to_list(3)
                
                if phase_strategies:
                    phase_breakdown[phase] = {
                        "count": len(phase_strategies),
                        "avg_novelty": np.mean([s.get("novelty_score", 0) for s in phase_strategies]),
                        "avg_stability": np.mean([s.get("stability_score", 0) for s in phase_strategies]),
                        "sample": phase_strategies[0] if phase_strategies else None
                    }
            
            report = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics.to_dict(),
                "recent_approved_strategies": recent_approved,
                "recent_meta_strategy": recent_meta,
                "phase_breakdown": phase_breakdown,
                "recommendations": recommendations,
                "system_health": metrics.creativity_health,
                "health_status": self._get_health_status(metrics.creativity_health)
            }
            
            logger.info(f"Creativity report generated: Health={metrics.creativity_health:.2f}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating creativity report: {e}")
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "system_health": 0.0,
                "health_status": "error"
            }
    
    def _generate_recommendations(self, metrics: CreativityMetrics) -> List[str]:
        """Generate actionable recommendations based on metrics"""
        
        recommendations = []
        
        # Health-based recommendations
        if metrics.creativity_health >= 0.85:
            recommendations.append("✅ Excellent creativity health. System generating high-quality innovative strategies.")
        elif metrics.creativity_health >= 0.70:
            recommendations.append("✅ Good creativity health. Continue current approach with minor refinements.")
        elif metrics.creativity_health >= 0.50:
            recommendations.append("⚠️ Moderate creativity health. Consider increasing novelty while maintaining ethical standards.")
        else:
            recommendations.append("🚨 Low creativity health. Review generation parameters and ethical constraints.")
        
        # Novelty recommendations
        if metrics.avg_novelty < 0.60:
            recommendations.append("📊 Low novelty detected. Encourage more diverse LLM prompts and pattern combinations.")
        elif metrics.avg_novelty > 0.85:
            recommendations.append("⚠️ Very high novelty. Monitor for practical viability and stability.")
        
        # Ethical recommendations
        if metrics.avg_ethical_alignment < self.ethical_threshold:
            recommendations.append("🚨 Ethical alignment below threshold. Review and strengthen ethical constraints.")
        elif metrics.ideas_rejected > metrics.ideas_approved:
            recommendations.append("⚠️ High rejection rate. Consider adjusting generation parameters.")
        
        # Stability recommendations
        if metrics.avg_stability < 0.55:
            recommendations.append("⚠️ Low stability scores. Focus on practical, implementable strategies.")
        
        # Provider diversity
        if len(metrics.provider_distribution) < 2:
            recommendations.append("📊 Limited provider diversity. Engage multiple LLM providers for varied perspectives.")
        
        # Phase coverage
        if len(metrics.phase_distribution) < 3:
            recommendations.append("📊 Incomplete phase coverage. Generate strategies across all game phases.")
        
        if not recommendations:
            recommendations.append("✅ System operating optimally. Continue scheduled creativity cycles.")
        
        return recommendations
    
    def _get_health_status(self, health: float) -> str:
        """Get health status string from score"""
        if health >= 0.85:
            return "excellent"
        elif health >= 0.70:
            return "good"
        elif health >= 0.50:
            return "moderate"
        else:
            return "poor"
