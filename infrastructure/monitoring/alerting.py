"""
Alerting and Notifications

This module provides comprehensive alerting and notification capabilities
with support for multiple channels, rule evaluation, and alert deduplication.
"""

import asyncio
import time
import json
import smtplib
import aiohttp
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
import threading
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    FIRING = "firing"
    RESOLVED = "resolved"
    SILENCED = "silenced"


class NotificationChannel(Enum):
    """Notification channels."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    CONSOLE = "console"


@dataclass
class Alert:
    """Alert representation."""
    name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    starts_at: float = field(default_factory=time.time)
    ends_at: Optional[float] = None
    fingerprint: str = ""
    generator_url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "labels": self.labels,
            "annotations": self.annotations,
            "starts_at": self.starts_at,
            "ends_at": self.ends_at,
            "fingerprint": self.fingerprint,
            "generator_url": self.generator_url
        }


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    condition: str  # Evaluation condition
    severity: AlertSeverity
    message: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    evaluation_interval: int = 60  # seconds
    for_duration: int = 0  # seconds to wait before firing
    channels: List[NotificationChannel] = field(default_factory=list)


@dataclass
class Silence:
    """Alert silence rule."""
    id: str
    matchers: Dict[str, str]
    starts_at: float
    ends_at: float
    created_by: str = "system"
    comment: str = ""

    def matches(self, alert: Alert) -> bool:
        """Check if silence matches alert."""
        for key, value in self.matchers.items():
            if alert.labels.get(key) != value:
                return False
        return True

    def is_active(self) -> bool:
        """Check if silence is currently active."""
        now = time.time()
        return self.starts_at <= now <= self.ends_at


class NotificationProvider:
    """Base class for notification providers."""

    async def send_alert(self, alert: Alert, channel_config: Dict[str, Any]) -> bool:
        """
        Send alert notification.

        Args:
            alert: Alert to send
            channel_config: Channel configuration

        Returns:
            True if notification sent successfully
        """
        raise NotImplementedError


class EmailNotificationProvider(NotificationProvider):
    """Email notification provider."""

    async def send_alert(self, alert: Alert, channel_config: Dict[str, Any]) -> bool:
        """Send alert via email."""
        try:
            smtp_host = channel_config.get("smtp_host")
            smtp_port = channel_config.get("smtp_port", 587)
            username = channel_config.get("username")
            password = channel_config.get("password")
            from_email = channel_config.get("from_email")
            to_emails = channel_config.get("to_emails", [])

            if not all([smtp_host, username, password, from_email, to_emails]):
                return False

            # Create email message
            msg = MimeMultipart()
            msg['From'] = from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.name}"

            # Create email body
            body = self._format_alert_email(alert)
            msg.attach(MimeText(body, 'html'))

            # Send email
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)

            return True

        except Exception as e:
            print(f"Failed to send email alert: {e}")
            return False

    def _format_alert_email(self, alert: Alert) -> str:
        """Format alert as HTML email."""
        severity_colors = {
            AlertSeverity.INFO: "#17a2b8",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.ERROR: "#fd7e14",
            AlertSeverity.CRITICAL: "#dc3545"
        }

        color = severity_colors.get(alert.severity, "#6c757d")

        html = f"""
        <html>
        <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background-color: {color}; color: white; padding: 20px; border-radius: 5px 5px 0 0;">
                    <h2 style="margin: 0;">{alert.name}</h2>
                    <p style="margin: 5px 0 0 0;">Severity: {alert.severity.value.upper()}</p>
                </div>

                <div style="background-color: #f8f9fa; padding: 20px; border: 1px solid #dee2e6; border-top: none;">
                    <h3 style="margin-top: 0;">Alert Details</h3>
                    <p><strong>Message:</strong> {alert.message}</p>
                    <p><strong>Status:</strong> {alert.status.value}</p>
                    <p><strong>Started:</strong> {datetime.fromtimestamp(alert.starts_at).strftime('%Y-%m-%d %H:%M:%S')}</p>
                    {f'<p><strong>Resolved:</strong> {datetime.fromtimestamp(alert.ends_at).strftime("%Y-%m-%d %H:%M:%S")}</p>' if alert.ends_at else ''}
                </div>

                {self._format_labels_section(alert.labels) if alert.labels else ''}
                {self._format_annotations_section(alert.annotations) if alert.annotations else ''}
            </div>
        </body>
        </html>
        """
        return html

    def _format_labels_section(self, labels: Dict[str, str]) -> str:
        """Format labels section."""
        return f"""
        <div style="background-color: white; padding: 20px; border: 1px solid #dee2e6; border-top: none;">
            <h4>Labels</h4>
            <table style="width: 100%; border-collapse: collapse;">
                {''.join(f'<tr><td style="padding: 5px; border: 1px solid #dee2e6;"><strong>{k}</strong></td><td style="padding: 5px; border: 1px solid #dee2e6;">{v}</td></tr>' for k, v in labels.items())}
            </table>
        </div>
        """

    def _format_annotations_section(self, annotations: Dict[str, str]) -> str:
        """Format annotations section."""
        return f"""
        <div style="background-color: white; padding: 20px; border: 1px solid #dee2e6; border-top: none; border-radius: 0 0 5px 5px;">
            <h4>Annotations</h4>
            <table style="width: 100%; border-collapse: collapse;">
                {''.join(f'<tr><td style="padding: 5px; border: 1px solid #dee2e6;"><strong>{k}</strong></td><td style="padding: 5px; border: 1px solid #dee2e6;">{v}</td></tr>' for k, v in annotations.items())}
            </table>
        </div>
        """


class WebhookNotificationProvider(NotificationProvider):
    """Webhook notification provider."""

    async def send_alert(self, alert: Alert, channel_config: Dict[str, Any]) -> bool:
        """Send alert via webhook."""
        try:
            url = channel_config.get("url")
            method = channel_config.get("method", "POST")
            headers = channel_config.get("headers", {})
            timeout = channel_config.get("timeout", 10)

            if not url:
                return False

            # Prepare payload
            payload = {
                "alert": alert.to_dict(),
                "timestamp": time.time(),
                "channel": "webhook"
            }

            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method,
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    return response.status < 400

        except Exception as e:
            print(f"Failed to send webhook alert: {e}")
            return False


class SlackNotificationProvider(NotificationProvider):
    """Slack notification provider."""

    async def send_alert(self, alert: Alert, channel_config: Dict[str, Any]) -> bool:
        """Send alert to Slack."""
        try:
            webhook_url = channel_config.get("webhook_url")
            channel = channel_config.get("channel", "#alerts")

            if not webhook_url:
                return False

            # Prepare Slack message
            color = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.ERROR: "danger",
                AlertSeverity.CRITICAL: "danger"
            }.get(alert.severity, "warning")

            payload = {
                "channel": channel,
                "username": "Alert Manager",
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color,
                        "title": f"{alert.severity.value.upper()}: {alert.name}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Status",
                                "value": alert.status.value,
                                "short": True
                            },
                            {
                                "title": "Started",
                                "value": datetime.fromtimestamp(alert.starts_at).strftime('%Y-%m-%d %H:%M:%S'),
                                "short": True
                            }
                        ],
                        "footer": "Alert Manager",
                        "ts": int(time.time())
                    }
                ]
            }

            # Add labels as fields
            if alert.labels:
                for key, value in alert.labels.items():
                    payload["attachments"][0]["fields"].append({
                        "title": key,
                        "value": value,
                        "short": True
                    })

            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    return response.status == 200

        except Exception as e:
            print(f"Failed to send Slack alert: {e}")
            return False


class ConsoleNotificationProvider(NotificationProvider):
    """Console notification provider for debugging."""

    async def send_alert(self, alert: Alert, channel_config: Dict[str, Any]) -> bool:
        """Print alert to console."""
        timestamp = datetime.fromtimestamp(alert.starts_at).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {alert.severity.value.upper()}: {alert.name}")
        print(f"  Status: {alert.status.value}")
        print(f"  Message: {alert.message}")
        if alert.labels:
            print(f"  Labels: {alert.labels}")
        print()
        return True


class AlertRuleEvaluator:
    """Alert rule evaluator."""

    def __init__(self):
        """Initialize evaluator."""
        self._functions = {
            "gt": lambda a, b: a > b,
            "lt": lambda a, b: a < b,
            "eq": lambda a, b: a == b,
            "ne": lambda a, b: a != b,
            "gte": lambda a, b: a >= b,
            "lte": lambda a, b: a <= b,
            "abs": lambda a: abs(a),
            "len": lambda a: len(a),
            "sum": lambda a: sum(a),
            "avg": lambda a: sum(a) / len(a) if a else 0,
            "max": lambda a: max(a) if a else 0,
            "min": lambda a: min(a) if a else 0,
        }

    def evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate alert condition.

        Args:
            condition: Condition expression
            context: Evaluation context

        Returns:
            True if condition is met
        """
        try:
            # Create safe evaluation environment
            safe_dict = {
                "__builtins__": {},
                **self._functions,
                **context
            }

            return eval(condition, safe_dict)
        except Exception as e:
            print(f"Failed to evaluate condition '{condition}': {e}")
            return False


class AlertManager:
    """
    Central alert manager.
    """

    def __init__(self):
        """Initialize alert manager."""
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.silences: Dict[str, Silence] = {}
        self.providers: Dict[NotificationChannel, NotificationProvider] = {
            NotificationChannel.EMAIL: EmailNotificationProvider(),
            NotificationChannel.WEBHOOK: WebhookNotificationProvider(),
            NotificationChannel.SLACK: SlackNotificationProvider(),
            NotificationChannel.CONSOLE: ConsoleNotificationProvider(),
        }
        self.channel_configs: Dict[NotificationChannel, Dict[str, Any]] = {}
        self.evaluator = AlertRuleEvaluator()
        self._evaluation_task: Optional[asyncio.Task] = None
        self._stop_evaluation = False

    def add_rule(self, rule: AlertRule) -> None:
        """
        Add alert rule.

        Args:
            rule: Alert rule to add
        """
        self.rules[rule.name] = rule

    def remove_rule(self, name: str) -> bool:
        """
        Remove alert rule.

        Args:
            name: Rule name

        Returns:
            True if rule was removed
        """
        if name in self.rules:
            del self.rules[name]
            return True
        return False

    def configure_channel(self, channel: NotificationChannel, config: Dict[str, Any]) -> None:
        """
        Configure notification channel.

        Args:
            channel: Channel type
            config: Channel configuration
        """
        self.channel_configs[channel] = config

    def add_silence(self, silence: Silence) -> None:
        """
        Add silence rule.

        Args:
            silence: Silence rule
        """
        self.silences[silence.id] = silence

    def remove_silence(self, silence_id: str) -> bool:
        """
        Remove silence rule.

        Args:
            silence_id: Silence ID

        Returns:
            True if silence was removed
        """
        if silence_id in self.silences:
            del self.silences[silence_id]
            return True
        return False

    def start_evaluation(self, interval: int = 60) -> None:
        """
        Start rule evaluation loop.

        Args:
            interval: Evaluation interval in seconds
        """
        if self._evaluation_task and not self._evaluation_task.done():
            return

        async def evaluation_loop():
            while not self._stop_evaluation:
                try:
                    await self._evaluate_rules()
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"Error in alert evaluation: {e}")

        self._evaluation_task = asyncio.create_task(evaluation_loop())

    def stop_evaluation(self) -> None:
        """Stop rule evaluation loop."""
        self._stop_evaluation = True
        if self._evaluation_task and not self._evaluation_task.done():
            self._evaluation_task.cancel()
            try:
                asyncio.get_event_loop().run_until_complete(self._evaluation_task)
            except asyncio.CancelledError:
                pass

    async def _evaluate_rules(self) -> None:
        """Evaluate all alert rules."""
        for rule in self.rules.values():
            if not rule.enabled:
                continue

            try:
                # Get context for evaluation
                context = await self._get_evaluation_context(rule)

                # Evaluate condition
                should_fire = self.evaluator.evaluate_condition(rule.condition, context)

                # Get existing alert
                existing_alert = self.active_alerts.get(rule.name)

                if should_fire:
                    await self._handle_firing_alert(rule, existing_alert)
                elif existing_alert:
                    await self._handle_resolved_alert(rule, existing_alert)

            except Exception as e:
                print(f"Error evaluating rule {rule.name}: {e}")

    async def _get_evaluation_context(self, rule: AlertRule) -> Dict[str, Any]:
        """
        Get evaluation context for rule.

        Args:
            rule: Alert rule

        Returns:
            Evaluation context
        """
        # This is a placeholder - in real implementation, this would
        # collect metrics from monitoring systems
        context = {
            "time": time.time(),
            "rule_name": rule.name,
        }

        # Add common metrics
        try:
            # Import here to avoid circular imports
            from .metrics import get_registry
            registry = get_registry()

            # Add all metric values to context
            for metric in registry.get_all_metrics().values():
                context[metric.name] = metric.get_value()

        except ImportError:
            pass

        return context

    async def _handle_firing_alert(self, rule: AlertRule, existing_alert: Optional[Alert]) -> None:
        """
        Handle firing alert.

        Args:
            rule: Alert rule
            existing_alert: Existing alert if any
        """
        now = time.time()

        if existing_alert:
            # Alert already firing, check if we should send notifications
            if existing_alert.status == AlertStatus.FIRING:
                # Only send notifications if configured interval has passed
                last_notification = existing_alert.annotations.get("last_notification", 0)
                if now - last_notification > rule.evaluation_interval:
                    await self._send_alert_notifications(existing_alert, rule)
                    existing_alert.annotations["last_notification"] = str(now)
        else:
            # New alert
            fingerprint = self._generate_fingerprint(rule)
            alert = Alert(
                name=rule.name,
                severity=rule.severity,
                status=AlertStatus.FIRING,
                message=rule.message,
                labels=rule.labels.copy(),
                annotations=rule.annotations.copy(),
                fingerprint=fingerprint
            )

            # Check silences
            if not self._is_silenced(alert):
                if rule.for_duration > 0:
                    # Wait for duration before firing
                    alert.annotations["pending_since"] = str(now)
                    self.active_alerts[rule.name] = alert
                else:
                    # Fire immediately
                    self.active_alerts[rule.name] = alert
                    await self._send_alert_notifications(alert, rule)

    async def _handle_resolved_alert(self, rule: AlertRule, existing_alert: Alert) -> None:
        """
        Handle resolved alert.

        Args:
            rule: Alert rule
            existing_alert: Existing alert
        """
        existing_alert.status = AlertStatus.RESOLVED
        existing_alert.ends_at = time.time()

        # Send resolution notification
        await self._send_alert_notifications(existing_alert, rule)

        # Remove from active alerts
        self.active_alerts.pop(rule.name, None)

    async def _send_alert_notifications(self, alert: Alert, rule: AlertRule) -> None:
        """
        Send alert notifications.

        Args:
            alert: Alert to send
            rule: Alert rule
        """
        # Check if alert is silenced
        if self._is_silenced(alert):
            return

        # Send to configured channels
        for channel in rule.channels:
            provider = self.providers.get(channel)
            config = self.channel_configs.get(channel, {})

            if provider and config:
                try:
                    await provider.send_alert(alert, config)
                except Exception as e:
                    print(f"Failed to send {channel.value} alert: {e}")

    def _is_silenced(self, alert: Alert) -> bool:
        """
        Check if alert is silenced.

        Args:
            alert: Alert to check

        Returns:
            True if alert is silenced
        """
        for silence in self.silences.values():
            if silence.is_active() and silence.matches(alert):
                return True
        return False

    def _generate_fingerprint(self, rule: AlertRule) -> str:
        """
        Generate alert fingerprint.

        Args:
            rule: Alert rule

        Returns:
            Fingerprint string
        """
        import hashlib
        content = f"{rule.name}:{rule.severity.value}:{sorted(rule.labels.items())}"
        return hashlib.md5(content.encode()).hexdigest()

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alert_status(self) -> Dict[str, Any]:
        """
        Get alert manager status.

        Returns:
            Status dictionary
        """
        active_count = len(self.active_alerts)
        firing_count = len([a for a in self.active_alerts.values() if a.status == AlertStatus.FIRING])
        silenced_count = len([s for s in self.silences.values() if s.is_active()])

        return {
            "active_alerts": active_count,
            "firing_alerts": firing_count,
            "silenced_rules": silenced_count,
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules.values() if r.enabled])
        }


# Global alert manager instance
_global_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = AlertManager()
    return _global_alert_manager


def set_alert_manager(manager: AlertManager) -> None:
    """Set global alert manager instance."""
    global _global_alert_manager
    _global_alert_manager = manager


# Export
__all__ = [
    'AlertSeverity',
    'AlertStatus',
    'NotificationChannel',
    'Alert',
    'AlertRule',
    'Silence',
    'NotificationProvider',
    'EmailNotificationProvider',
    'WebhookNotificationProvider',
    'SlackNotificationProvider',
    'ConsoleNotificationProvider',
    'AlertRuleEvaluator',
    'AlertManager',
    'get_alert_manager',
    'set_alert_manager'
]