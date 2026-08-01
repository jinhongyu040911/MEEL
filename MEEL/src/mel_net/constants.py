from __future__ import annotations

EDIT_OPERATIONS = [
    "identity",
    "image_swap",
    "entity_shift",
    "relation_misbind",
    "context_drop",
    "text_overclaim",
]

MANIPULATION_OPERATIONS = EDIT_OPERATIONS[1:]

DEFAULT_HOLDOUT_COMPOSITIONS = [
    "image_swap+text_overclaim",
    "entity_shift+relation_misbind",
    "context_drop+text_overclaim",
]

