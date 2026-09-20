# Governance

Prosody Markup starts with a BDFL model. A small specification committee becomes appropriate after
three independent implementations exist.

## Change classes

- **Threshold/config changes:** cheap, provided versioned eval results do not regress precision.
- **IR additions:** additive when old consumers can ignore them; otherwise require schema major.
- **Renderer changes:** must preserve legend semantics and compatibility snapshots.
- **Tier 1 additions/reassignments:** expensive and require discrimination data.

No Tier 1 mark is added because it merely feels intuitive. A proposal must include human labels,
precision results, acquisition or pilot discrimination results, collision analysis with existing
writing conventions, and a migration/versioning plan.

Outputs always declare their legend version. Readers trained on one major legend must not silently
receive another.
