# Archived datasets

These files are superseded by the active pipeline and are kept here for reference only.
None of them feed into `train_synthetic.jsonl`.

| File | Records | Reason archived |
|---|---|---|
| `synthetic_kaya_v7.jsonl` | 5,142 | 100% content overlap with `synthetic_kaya.jsonl` at time of archive |
| `synthetic_kaya_v8.jsonl` | 5,974 | 100% content overlap with `synthetic_kaya.jsonl` at time of archive |
| `synthetic_kaya_azure.jsonl` | 105 | Azure-generated examples, never merged into active pipeline |
| `synthetic_kaya_xai.jsonl` | 105 | xAI-generated examples, never merged into active pipeline |
| `synthetic_kaya.jsonl.bak` | — | Manual backup, superseded |
| `targeted_qa_draft.jsonl` | 240 | Unfiltered draft — superseded by `targeted_qa_draft_filtered.jsonl` |
| `targeted_qa_draft_synth_v6.jsonl` | 45 | v6 draft — superseded by v7/v8 |
| `targeted_qa_draft_synth_v7_dirty.jsonl` | 180 | Dirty draft — superseded by `targeted_qa_v8.jsonl` |
| `targeted_qa_draft_filtered.jsonl` | 239 | Filtered draft — superseded by `targeted_qa_v8.jsonl` |

## Active files (in `data/`)

| File | Purpose |
|---|---|
| `synthetic_kaya.jsonl` | Primary input to `merge_datasets.py` — output of `format_direct_training.py` + targeted Q&A |
| `targeted_qa_v8.jsonl` | Latest targeted Q&A (165 records; 79 not yet in `synthetic_kaya.jsonl`) |
| `synthetic_portuguese.jsonl` | General PT instruction data — available but not currently merged |
| `train_synthetic.jsonl` | Final training set (output of `merge_datasets.py`) |
| `val_synthetic.jsonl` | Final validation set (output of `merge_datasets.py`) |
| `all_messages_cleaned.jsonl` | Extracted + cleaned chat messages (pipeline intermediate) |
| `finetune_chunks.jsonl` | 50K-token chunks for API-based generation (pipeline intermediate) |
