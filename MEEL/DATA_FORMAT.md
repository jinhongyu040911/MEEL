# Precomputed Feature Format

MEEL trains on frozen multimodal representations. Every sample is serialized
with `torch.save` and must provide the following fields:

| Field | Type and shape | Description |
|---|---|---|
| `sample_id` | string | Stable sample identifier |
| `label` | integer | Binary veracity label (`0` or `1`) |
| `clip_text_global` | float tensor `[D_t]` | Global text representation |
| `clip_image_global` | float tensor `[D_i]` | Global image representation |
| `text_entity_features` | float tensor `[N_t, D_e]` | Text-entity representations |
| `image_entity_features` | float tensor `[N_i, D_e]` | Image-region representations |
| `text_entity_mask` | float tensor `[N_t]` | Valid text-entity positions |
| `image_entity_mask` | float tensor `[N_i]` | Valid image-region positions |

Optional fields are `text` or `caption`, `dataset`, and `split`. Global text
and image dimensions may differ, while text-entity and image-region features
must share the same final dimension. Samples within one dataset must use
consistent dimensions and padded entity counts.

A minimal sample has the following structure:

```python
sample = {
    "sample_id": "example-0001",
    "label": 1,
    "clip_text_global": text_global,
    "clip_image_global": image_global,
    "text_entity_features": text_entities,
    "image_entity_features": image_regions,
    "text_entity_mask": text_mask,
    "image_entity_mask": image_mask,
}
```

Place individual sample files under `<feature-root>/<dataset>/<split>/*.pt`,
or save a list of sample dictionaries as `<feature-root>/<dataset>/<split>.pt`.
The current training entry point uses split directories by default.
