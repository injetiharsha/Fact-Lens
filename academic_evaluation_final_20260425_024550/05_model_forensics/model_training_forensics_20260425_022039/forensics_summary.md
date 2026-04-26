# Model Training Forensics Summary

- Training configs found: **9**
- Checkpoint run summaries found: **9**
- Dataset summaries found: **11**
- Detected fine-tune lineage edges: **2**
- Prior/extra checkpoints on `W:` (config.json count): **7**

## Configs by Task
- checkability: 1
- context: 3
- relevance: 1
- stance: 4

## Detected Lineage Edges
- stance_en_deberta_v1_mnli -> stance_en_deberta_v1_fever (fine_tune_from_checkpoint)
- stance_en_deberta_v1_fever -> stance_en_deberta_v1_vitaminc (fine_tune_from_checkpoint)