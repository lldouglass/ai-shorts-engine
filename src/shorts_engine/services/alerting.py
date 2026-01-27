"""Alerting service for pipeline failures and QA issues."""

import smtplib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import StrEnum
from typing import Any
from uuid import UUID

import httpx

from shorts_engine.config import settings
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


class AlertSeverity(StrEnum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """An alert to be sent via configured channels."""

    title: str
    message: str
    severity: AlertSeverity = AlertSeverity.ERROR
    context: dict[str, Any] = field(default_factory=dict)
    video_job_id: UUID | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class AlertingService:
    """Service for sending alerts via Discord and email.

    Supports:
    - Discord webhooks for real-time notifications
    - Email via SMTP for critical alerts
    """

    # Discord embed colors by severity
    DISCORD_COLORS = {
        AlertSeverity.INFO: 0x3498DB,  # Blue
        AlertSeverity.WARNING: 0xF39C12,  # Orange
        AlertSeverity.ERROR: 0xE74C3C,  # Red
        AlertSeverity.CRITICAL: 0x9B59B6,  # Purple
    }

    def __init__(self) -> None:
        """Initialize alerting service."""
        self.discord_webhook_url = settings.alert_discord_webhook_url
        self.email_enabled = bool(
            settings.alert_email_smtp_host and settings.alert_email_from and settings.alert_email_to
        )

    async def send_alert(self, alert: Alert) -> bool:
        """Send an alert via all configured channels.

        Args:
            alert: The alert to send

        Returns:
            True if at least one channel succeeded
        """
        results = []

        if self.discord_webhook_url:
            try:
                success = await self._send_discord(alert)
                results.append(success)
            except Exception as e:
                logger.error("discord_alert_failed", error=str(e))
                results.append(False)

        if self.email_enabled:
            try:
                success = self._send_email(alert)
                results.append(success)
            except Exception as e:
                logger.error("email_alert_failed", error=str(e))
                results.append(False)

        return any(results) if results else False

    async def _send_discord(self, alert: Alert) -> bool:
        """Send alert to Discord webhook.

        Args:
            alert: The alert to send

        Returns:
            True if successful
        """
        if not self.discord_webhook_url:
            return False

        # Build embed fields from context
        fields = []
        if alert.video_job_id:
            fields.append(
                {
                    "name": "Video Job ID",
                    "value": f"`{alert.video_job_id}`",
                    "inline": True,
                }
            )
        for key, value in alert.context.items():
            # Truncate long values
            str_value = str(value)
            if len(str_value) > 200:
                str_value = str_value[:197] + "..."
            fields.append(
                {
                    "name": key.replace("_", " ").title(),
                    "value": str_value,
                    "inline": True,
                }
            )

        payload = {
            "embeds": [
                {
                    "title": f"[{alert.severity.value.upper()}] {alert.title}",
                    "description": alert.message,
                    "color": self.DISCORD_COLORS.get(alert.severity, 0xE74C3C),
                    "fields": fields[:25],  # Discord limit
                    "timestamp": alert.timestamp.isoformat(),
                    "footer": {
                        "text": "AI Shorts Engine Alerting",
                    },
                }
            ]
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.discord_webhook_url,
                json=payload,
                timeout=10.0,
            )
            response.raise_for_status()

        logger.info(
            "discord_alert_sent",
            severity=alert.severity.value,
            title=alert.title,
        )
        return True

    def _send_email(self, alert: Alert) -> bool:
        """Send alert via email.

        Args:
            alert: The alert to send

        Returns:
            True if successful
        """
        if not self.email_enabled:
            return False

        # Build email content
        subject = f"[{alert.severity.value.upper()}] {alert.title}"

        # HTML body
        context_html = ""
        if alert.context:
            context_html = "<h3>Context</h3><table border='1' cellpadding='5'>"
            for key, value in alert.context.items():
                context_html += f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>"
            context_html += "</table>"

        html_body = f"""
        <html>
        <body>
        <h2>{alert.title}</h2>
        <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
        <p><strong>Time:</strong> {alert.timestamp.isoformat()}</p>
        {f"<p><strong>Video Job ID:</strong> {alert.video_job_id}</p>" if alert.video_job_id else ""}
        <h3>Message</h3>
        <p>{alert.message}</p>
        {context_html}
        <hr>
        <p><em>Sent by AI Shorts Engine Alerting</em></p>
        </body>
        </html>
        """

        # Plain text fallback
        text_body = f"""
{alert.title}
Severity: {alert.severity.value.upper()}
Time: {alert.timestamp.isoformat()}
{"Video Job ID: " + str(alert.video_job_id) if alert.video_job_id else ""}

Message:
{alert.message}

Context:
{chr(10).join(f"  {k}: {v}" for k, v in alert.context.items())}

--
Sent by AI Shorts Engine Alerting
        """

        # Validate required email settings (should already be checked by email_enabled)
        if not settings.alert_email_from or not settings.alert_email_smtp_host:
            logger.warning("email_settings_missing")
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = settings.alert_email_from
        msg["To"] = ", ".join(settings.alert_email_to)

        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        # Send via SMTP
        with smtplib.SMTP(
            settings.alert_email_smtp_host,
            settings.alert_email_smtp_port,
        ) as server:
            server.starttls()
            if settings.alert_email_username and settings.alert_email_password:
                server.login(
                    settings.alert_email_username,
                    settings.alert_email_password,
                )
            server.sendmail(
                settings.alert_email_from,
                settings.alert_email_to,
                msg.as_string(),
            )

        logger.info(
            "email_alert_sent",
            severity=alert.severity.value,
            title=alert.title,
            recipients=len(settings.alert_email_to),
        )
        return True


# Singleton instance
_alerting_service: AlertingService | None = None


def get_alerting_service() -> AlertingService:
    """Get the alerting service singleton."""
    global _alerting_service
    if _alerting_service is None:
        _alerting_service = AlertingService()
    return _alerting_service


# Convenience functions
async def alert_qa_failure(
    video_job_id: UUID,
    qa_stage: str,
    feedback: str,
    attempt: int,
    max_attempts: int,
    scores: dict[str, float] | None = None,
) -> None:
    """Send an alert for QA failure.

    Args:
        video_job_id: The video job ID
        qa_stage: Stage where QA failed (post_planning, post_render)
        feedback: QA feedback message
        attempt: Current attempt number
        max_attempts: Maximum allowed attempts
        scores: Optional QA scores dict
    """
    if not settings.alert_on_qa_failure:
        return

    is_final = attempt >= max_attempts
    severity = AlertSeverity.ERROR if is_final else AlertSeverity.WARNING

    context: dict[str, Any] = {
        "stage": qa_stage,
        "attempt": f"{attempt}/{max_attempts}",
    }
    if scores:
        context.update(scores)

    alert = Alert(
        title=f"QA {'Failed Permanently' if is_final else 'Check Failed'}",
        message=feedback,
        severity=severity,
        context=context,
        video_job_id=video_job_id,
    )

    service = get_alerting_service()
    await service.send_alert(alert)


async def alert_pipeline_failure(
    video_job_id: UUID,
    stage: str,
    error_message: str,
    error_type: str = "unknown",
) -> None:
    """Send an alert for pipeline failure.

    Args:
        video_job_id: The video job ID
        stage: Pipeline stage where failure occurred
        error_message: Error message
        error_type: Type of error (e.g., "timeout", "api_error")
    """
    if not settings.alert_on_pipeline_failure:
        return

    alert = Alert(
        title=f"Pipeline Failed at {stage}",
        message=error_message,
        severity=AlertSeverity.ERROR,
        context={
            "stage": stage,
            "error_type": error_type,
        },
        video_job_id=video_job_id,
    )

    service = get_alerting_service()
    await service.send_alert(alert)


async def alert_high_failure_rate(
    stage: str,
    failure_rate: float,
    window_hours: int = 1,
) -> None:
    """Send an alert for high failure rate.

    Args:
        stage: Pipeline stage with high failure rate
        failure_rate: Current failure rate (0.0-1.0)
        window_hours: Time window in hours
    """
    alert = Alert(
        title=f"High Failure Rate in {stage}",
        message=f"Failure rate of {failure_rate:.1%} detected over the last {window_hours} hour(s).",
        severity=AlertSeverity.CRITICAL if failure_rate > 0.3 else AlertSeverity.WARNING,
        context={
            "stage": stage,
            "failure_rate": f"{failure_rate:.1%}",
            "window_hours": window_hours,
        },
    )

    service = get_alerting_service()
    await service.send_alert(alert)


async def alert_queue_depth_high(
    stage: str,
    depth: int,
    threshold: int = 50,
) -> None:
    """Send an alert for high queue depth.

    Args:
        stage: Pipeline stage with high queue depth
        depth: Current queue depth
        threshold: Threshold that was exceeded
    """
    alert = Alert(
        title=f"High Queue Depth in {stage}",
        message=f"Queue depth of {depth} jobs exceeds threshold of {threshold}.",
        severity=AlertSeverity.WARNING,
        context={
            "stage": stage,
            "queue_depth": depth,
            "threshold": threshold,
        },
    )

    service = get_alerting_service()
    await service.send_alert(alert)
