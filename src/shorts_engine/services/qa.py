"""QA validation service for video content quality gates."""

import json
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from shorts_engine.adapters.llm.anthropic import AnthropicProvider
from shorts_engine.adapters.llm.base import LLMMessage, LLMProvider
from shorts_engine.adapters.llm.openai import OpenAIProvider
from shorts_engine.adapters.llm.stub import StubLLMProvider
from shorts_engine.config import settings
from shorts_engine.db.models import QAResultModel, VideoJobModel
from shorts_engine.domain.enums import QACheckType, QAStage
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QAResult:
    """Result of a QA check."""

    passed: bool
    hook_clarity_score: float | None = None
    coherence_score: float | None = None
    uniqueness_score: float | None = None
    policy_passed: bool | None = None
    policy_violations: list[str] = field(default_factory=list)
    feedback: str = ""
    similar_job_id: UUID | None = None
    similarity_score: float | None = None
    raw_response: dict[str, Any] | None = None
    model_used: str | None = None
    duration_seconds: float | None = None


class QAFailedException(Exception):
    """Exception raised when QA check fails but can be retried."""

    def __init__(self, message: str, qa_result: QAResult):
        super().__init__(message)
        self.qa_result = qa_result


class QAService:
    """Service for quality assurance validation of video content.

    Performs LLM-based quality checks including:
    - Hook clarity: How compelling is the opening?
    - Coherence: How well does the narrative flow?
    - Policy compliance: Check for violations
    - Uniqueness: Compare against recent scripts
    """

    PLAN_QA_SYSTEM_PROMPT = """You are a content quality analyst for short-form video scripts.
Your job is to evaluate video plans for quality before they enter production.

You will receive a video plan with scenes, title, and description.

Evaluate the following criteria and return a JSON object:

{
    "hook_clarity_score": 0.0-1.0,
    "coherence_score": 0.0-1.0,
    "policy_passed": true/false,
    "policy_violations": ["list of specific violations if any"],
    "feedback": "Brief constructive feedback for improvement"
}

Scoring guidelines:

HOOK CLARITY (0.0-1.0):
- 0.0-0.3: Weak/confusing opening, no clear hook
- 0.4-0.6: Decent hook but could be stronger
- 0.7-0.8: Good hook that grabs attention
- 0.9-1.0: Excellent, irresistible hook

COHERENCE (0.0-1.0):
- 0.0-0.3: Disjointed, scenes don't connect well
- 0.4-0.6: Somewhat coherent but has gaps
- 0.7-0.8: Good narrative flow
- 0.9-1.0: Perfect story arc with excellent transitions

POLICY COMPLIANCE:
Check for these violations:
- Medical/health claims without disclaimers
- Financial advice or get-rich-quick schemes
- Harmful content (violence, self-harm, dangerous activities)
- Misleading information or fake news
- Content targeting minors inappropriately
- Hate speech or discrimination
- Copyright infringement indicators

Be strict but fair. Return policy_passed=false only for clear violations."""

    RENDER_QA_SYSTEM_PROMPT = """You are a final quality check analyst for video scripts.
This is the last check before a video is published.

Evaluate the final script for:
1. Overall quality and professionalism
2. Clear messaging without confusion
3. Appropriate pacing and flow
4. Policy compliance

Return a JSON object:
{
    "hook_clarity_score": 0.0-1.0,
    "coherence_score": 0.0-1.0,
    "policy_passed": true/false,
    "policy_violations": [],
    "feedback": "Brief assessment"
}

Be thorough but efficient. This check is a final gate."""

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        """Initialize QA service with an LLM provider."""
        self.llm = llm_provider or self._get_default_provider()
        logger.info("qa_service_initialized", provider=self.llm.name)

    def _get_default_provider(self) -> LLMProvider:
        """Get the default LLM provider based on available API keys."""
        if settings.openai_api_key:
            return OpenAIProvider()
        if settings.anthropic_api_key:
            return AnthropicProvider()
        logger.warning("No LLM API keys configured, using stub provider for QA")
        return StubLLMProvider()

    def _compute_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts (word-level)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def _extract_script_text(self, video_job: VideoJobModel) -> str:
        """Extract the full script text from a video job for comparison."""
        parts = []
        if video_job.title:
            parts.append(video_job.title)
        if video_job.description:
            parts.append(video_job.description)
        if video_job.idea:
            parts.append(video_job.idea)
        # Add scene caption beats
        for scene in sorted(video_job.scenes, key=lambda s: s.scene_number):
            if scene.caption_beat:
                parts.append(scene.caption_beat)
            if scene.visual_prompt:
                parts.append(scene.visual_prompt)
        return " ".join(parts)

    async def _check_uniqueness(
        self,
        video_job: VideoJobModel,
        session: Session,
    ) -> tuple[float, UUID | None, float]:
        """Check uniqueness against recent scripts from the same project.

        Returns:
            Tuple of (uniqueness_score, similar_job_id, similarity_score)
            uniqueness_score is 1.0 - max_similarity
        """
        if not settings.qa_enabled:
            return 1.0, None, 0.0

        current_script = self._extract_script_text(video_job)

        # Get recent video jobs from the same project
        recent_jobs = (
            session.query(VideoJobModel)
            .filter(
                VideoJobModel.project_id == video_job.project_id,
                VideoJobModel.id != video_job.id,
                VideoJobModel.stage.in_(["planned", "generating", "ready", "ready_to_publish"]),
            )
            .order_by(VideoJobModel.created_at.desc())
            .limit(settings.qa_uniqueness_lookback_count)
            .all()
        )

        max_similarity = 0.0
        most_similar_job_id = None

        for job in recent_jobs:
            job_script = self._extract_script_text(job)
            similarity = self._compute_jaccard_similarity(current_script, job_script)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_job_id = job.id

        uniqueness_score = 1.0 - max_similarity
        return uniqueness_score, most_similar_job_id, max_similarity

    async def check_plan(
        self,
        video_job: VideoJobModel,
        session: Session,
    ) -> QAResult:
        """Run QA checks on a video plan after planning stage.

        Args:
            video_job: The video job with plan data
            session: Database session for uniqueness check

        Returns:
            QAResult with scores and pass/fail status
        """
        if not settings.qa_enabled:
            logger.info("qa_skipped", reason="qa_disabled", video_job_id=str(video_job.id))
            return QAResult(passed=True, feedback="QA disabled")

        start_time = time.time()
        logger.info("qa_plan_check_started", video_job_id=str(video_job.id))

        # Build the plan content for LLM evaluation
        scenes_text = []
        for scene in sorted(video_job.scenes, key=lambda s: s.scene_number):
            scenes_text.append(
                f"Scene {scene.scene_number}:\n"
                f"  Visual: {scene.visual_prompt}\n"
                f"  Caption: {scene.caption_beat}\n"
                f"  Duration: {scene.duration_seconds}s"
            )

        plan_content = f"""Title: {video_job.title or 'Untitled'}
Description: {video_job.description or 'No description'}
Idea: {video_job.idea}
Style: {video_job.style_preset}

Scenes:
{chr(10).join(scenes_text)}"""

        # Get LLM evaluation
        messages = [
            LLMMessage(role="system", content=self.PLAN_QA_SYSTEM_PROMPT),
            LLMMessage(role="user", content=f"Evaluate this video plan:\n\n{plan_content}"),
        ]

        try:
            response = await self.llm.complete(
                messages=messages,
                temperature=0.3,  # Low temperature for consistent evaluation
                max_tokens=1024,
                json_mode=True,
            )
            eval_data = json.loads(response.content)
        except Exception as e:
            logger.error("qa_llm_error", error=str(e), video_job_id=str(video_job.id))
            # On LLM error, pass with warning
            return QAResult(
                passed=True,
                feedback=f"QA check skipped due to LLM error: {e}",
                model_used=self.llm.name,
                duration_seconds=time.time() - start_time,
            )

        # Check uniqueness
        uniqueness_score, similar_job_id, similarity_score = await self._check_uniqueness(
            video_job, session
        )

        # Determine pass/fail
        hook_clarity = eval_data.get("hook_clarity_score", 0.0)
        coherence = eval_data.get("coherence_score", 0.0)
        policy_passed = eval_data.get("policy_passed", True)
        policy_violations = eval_data.get("policy_violations", [])

        passed = (
            hook_clarity >= settings.qa_hook_clarity_threshold
            and coherence >= settings.qa_coherence_threshold
            and policy_passed
            and similarity_score < settings.qa_uniqueness_similarity_threshold
        )

        # Build feedback
        feedback_parts = []
        if eval_data.get("feedback"):
            feedback_parts.append(eval_data["feedback"])
        if hook_clarity < settings.qa_hook_clarity_threshold:
            feedback_parts.append(
                f"Hook clarity too low ({hook_clarity:.2f} < {settings.qa_hook_clarity_threshold})"
            )
        if coherence < settings.qa_coherence_threshold:
            feedback_parts.append(
                f"Coherence too low ({coherence:.2f} < {settings.qa_coherence_threshold})"
            )
        if not policy_passed:
            feedback_parts.append(f"Policy violations: {', '.join(policy_violations)}")
        if similarity_score >= settings.qa_uniqueness_similarity_threshold:
            feedback_parts.append(
                f"Too similar to existing content ({similarity_score:.2f} similarity)"
            )

        duration = time.time() - start_time
        result = QAResult(
            passed=passed,
            hook_clarity_score=hook_clarity,
            coherence_score=coherence,
            uniqueness_score=uniqueness_score,
            policy_passed=policy_passed,
            policy_violations=policy_violations,
            feedback=" | ".join(feedback_parts) if feedback_parts else "Passed all checks",
            similar_job_id=similar_job_id,
            similarity_score=similarity_score,
            raw_response=eval_data,
            model_used=self.llm.name,
            duration_seconds=duration,
        )

        logger.info(
            "qa_plan_check_completed",
            video_job_id=str(video_job.id),
            passed=passed,
            hook_clarity=hook_clarity,
            coherence=coherence,
            uniqueness=uniqueness_score,
            duration=duration,
        )

        return result

    async def check_render(
        self,
        video_job: VideoJobModel,
        _session: Session,
    ) -> QAResult:
        """Run QA checks after rendering (final script validation).

        Args:
            video_job: The video job after rendering
            session: Database session

        Returns:
            QAResult with scores and pass/fail status
        """
        if not settings.qa_enabled:
            logger.info("qa_skipped", reason="qa_disabled", video_job_id=str(video_job.id))
            return QAResult(passed=True, feedback="QA disabled")

        start_time = time.time()
        logger.info("qa_render_check_started", video_job_id=str(video_job.id))

        # Build the final script content
        scenes_text = []
        for scene in sorted(video_job.scenes, key=lambda s: s.scene_number):
            scenes_text.append(f"[Scene {scene.scene_number}] {scene.caption_beat}")

        script_content = f"""Title: {video_job.title or 'Untitled'}
Description: {video_job.description or 'No description'}

Script:
{chr(10).join(scenes_text)}"""

        # Get LLM evaluation
        messages = [
            LLMMessage(role="system", content=self.RENDER_QA_SYSTEM_PROMPT),
            LLMMessage(role="user", content=f"Final check for this video:\n\n{script_content}"),
        ]

        try:
            response = await self.llm.complete(
                messages=messages,
                temperature=0.2,
                max_tokens=512,
                json_mode=True,
            )
            eval_data = json.loads(response.content)
        except Exception as e:
            logger.error("qa_llm_error", error=str(e), video_job_id=str(video_job.id))
            return QAResult(
                passed=True,
                feedback=f"QA check skipped due to LLM error: {e}",
                model_used=self.llm.name,
                duration_seconds=time.time() - start_time,
            )

        # Determine pass/fail (more lenient for render QA)
        hook_clarity = eval_data.get("hook_clarity_score", 0.0)
        coherence = eval_data.get("coherence_score", 0.0)
        policy_passed = eval_data.get("policy_passed", True)
        policy_violations = eval_data.get("policy_violations", [])

        # Render QA is primarily about policy - if we got here, planning QA passed
        passed = policy_passed

        duration = time.time() - start_time
        result = QAResult(
            passed=passed,
            hook_clarity_score=hook_clarity,
            coherence_score=coherence,
            policy_passed=policy_passed,
            policy_violations=policy_violations,
            feedback=eval_data.get("feedback", ""),
            raw_response=eval_data,
            model_used=self.llm.name,
            duration_seconds=duration,
        )

        logger.info(
            "qa_render_check_completed",
            video_job_id=str(video_job.id),
            passed=passed,
            duration=duration,
        )

        return result

    def save_qa_result(
        self,
        session: Session,
        video_job_id: UUID,
        check_type: QACheckType,
        stage: QAStage,
        attempt_number: int,
        result: QAResult,
    ) -> QAResultModel:
        """Persist a QA result to the database.

        Args:
            session: Database session
            video_job_id: ID of the video job
            check_type: Type of QA check (plan_qa, render_qa)
            stage: Pipeline stage (post_planning, post_render)
            attempt_number: Which attempt this is (1, 2, 3...)
            result: The QA result to save

        Returns:
            The created QAResultModel
        """
        qa_record = QAResultModel(
            video_job_id=video_job_id,
            check_type=check_type.value,
            stage=stage.value,
            attempt_number=attempt_number,
            passed=result.passed,
            hook_clarity_score=result.hook_clarity_score,
            coherence_score=result.coherence_score,
            uniqueness_score=result.uniqueness_score,
            policy_passed=result.policy_passed,
            policy_violations=result.policy_violations,
            feedback=result.feedback,
            similar_job_id=result.similar_job_id,
            similarity_score=result.similarity_score,
            raw_response=result.raw_response,
            model_used=result.model_used,
            duration_seconds=result.duration_seconds,
        )
        session.add(qa_record)
        session.flush()

        logger.info(
            "qa_result_saved",
            qa_result_id=str(qa_record.id),
            video_job_id=str(video_job_id),
            check_type=check_type.value,
            passed=result.passed,
        )

        return qa_record


def get_qa_service() -> QAService:
    """Get a QA service instance."""
    return QAService()
