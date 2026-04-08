"""
FreshChain WhatsApp Alert Simulation
server/whatsapp_alerts.py

Simulates real-time WhatsApp alerts to farmers, transporters,
and mandi operators — just like the real FreshChain AI system.

In production this would use the WhatsApp Business API.
Here we simulate it and return alert objects for the dashboard.
"""

from datetime import datetime
from typing import List, Dict
from enum import Enum


class AlertType(str, Enum):
    SPOILAGE_WARNING   = "spoilage_warning"
    DISPATCH_CONFIRMED = "dispatch_confirmed"
    TRUCK_BREAKDOWN    = "truck_breakdown"
    PRICE_SPIKE        = "price_spike"
    PRICE_DROP         = "price_drop"
    EPISODE_START      = "episode_start"
    EPISODE_END        = "episode_end"
    REROUTE            = "reroute"
    CRITICAL_BATCH     = "critical_batch"


class AlertRecipient(str, Enum):
    FARMER     = "Farmer"
    TRANSPORTER = "Transporter"
    MANDI      = "Mandi Operator"
    MANAGER    = "Warehouse Manager"


# Message templates in English (production would add Hindi/Marathi)
ALERT_TEMPLATES = {
    AlertType.SPOILAGE_WARNING: {
        "recipient": AlertRecipient.FARMER,
        "emoji": "⚠️",
        "template": "ALERT: Batch {batch_id} ({crop_type}) spoilage risk is {risk}%. Immediate action required. Days in storage: {days}.",
    },
    AlertType.DISPATCH_CONFIRMED: {
        "recipient": AlertRecipient.TRANSPORTER,
        "emoji": "✅",
        "template": "CONFIRMED: {quantity}kg of {crop_type} (Batch {batch_id}) dispatched to {destination} via Truck {truck_id}. Est. value: ₹{value}.",
    },
    AlertType.TRUCK_BREAKDOWN: {
        "recipient": AlertRecipient.MANAGER,
        "emoji": "🚨",
        "template": "EMERGENCY: Truck {truck_id} breakdown reported. {batches_at_risk} batch(es) at risk. Arrange alternate transport immediately.",
    },
    AlertType.PRICE_SPIKE: {
        "recipient": AlertRecipient.MANDI,
        "emoji": "📈",
        "template": "MARKET UPDATE: {crop_type} price up to ₹{price}/kg at {market}. Good time to dispatch.",
    },
    AlertType.PRICE_DROP: {
        "recipient": AlertRecipient.FARMER,
        "emoji": "📉",
        "template": "MARKET UPDATE: {crop_type} price dropped to ₹{price}/kg. Consider holding if storage conditions allow.",
    },
    AlertType.CRITICAL_BATCH: {
        "recipient": AlertRecipient.MANAGER,
        "emoji": "🔴",
        "template": "CRITICAL: Batch {batch_id} ({crop_type}, {quantity}kg) spoilage risk exceeded 85%. Recommend immediate dispatch or discard.",
    },
    AlertType.EPISODE_START: {
        "recipient": AlertRecipient.MANAGER,
        "emoji": "🌾",
        "template": "FreshChain System Online. Task: {task_id}. {num_batches} batch(es) in storage. Total quantity: {total_kg}kg.",
    },
    AlertType.EPISODE_END: {
        "recipient": AlertRecipient.FARMER,
        "emoji": "📊",
        "template": "Episode Summary: Saved {saved_kg}kg | Lost {lost_kg}kg | Score: {score}/1.0. View full report at FreshChain dashboard.",
    },
    AlertType.REROUTE: {
        "recipient": AlertRecipient.TRANSPORTER,
        "emoji": "🔀",
        "template": "REROUTE: Batch {batch_id} redirected to {destination}. New market price: ₹{price}/kg.",
    },
}


class WhatsAppAlertSystem:
    """
    Simulates WhatsApp Business API alerts.
    Tracks all alerts for the current episode.
    """

    def __init__(self):
        self._alerts: List[Dict] = []

    def reset(self):
        self._alerts = []

    def send(self, alert_type: AlertType, **kwargs) -> Dict:
        """Generate and store an alert."""
        template_data = ALERT_TEMPLATES.get(alert_type, {})
        emoji = template_data.get("emoji", "📱")
        recipient = template_data.get("recipient", "User")
        template = template_data.get("template", "Alert: {}")

        try:
            message = template.format(**kwargs)
        except KeyError:
            message = f"{alert_type.value}: {kwargs}"

        alert = {
            "type": alert_type.value,
            "recipient": recipient.value if hasattr(recipient, 'value') else str(recipient),
            "emoji": emoji,
            "message": f"{emoji} *FreshChain Alert*\nTo: {recipient}\n\n{message}",
            "short": f"{emoji} {message[:80]}{'...' if len(message) > 80 else ''}",
            "timestamp": datetime.now().isoformat(),
            "delivered": True,  # Simulated delivery
        }
        self._alerts.append(alert)
        return alert

    def get_alerts(self) -> List[Dict]:
        return list(reversed(self._alerts))

    def get_recent(self, n: int = 5) -> List[Dict]:
        return list(reversed(self._alerts))[:n]


# Global alert system instance
alert_system = WhatsAppAlertSystem()

    def clear(self):
        """Clear all alerts (called on episode reset)."""
        self._alerts = []
