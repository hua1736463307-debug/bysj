"""Canonical behavior alarm categories and compatibility constants."""

SUPPORTED_EVENTS = (
    "phone_use",
    "missing_badge",
    "fall_down",
)

EVENT_NAME_MAP = {
    "phone_use": "\u4f7f\u7528\u624b\u673a",
    "missing_badge": "\u672a\u4f69\u6234\u5de5\u724c",
    "fall_down": "\u4eba\u5458\u6454\u5012",
}

DEFAULT_STAGE2_EVENTS = list(SUPPORTED_EVENTS)

# Compatibility name retained for callers that still import the old constant.
DEFAULT_TRANSFER_CLASSES = ["cell phone"]
