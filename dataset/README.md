# Dataset Label Format (MultiBypassT40)

We follow the JSON label format to create pytorch datasets. Each video has a corresponding JSON file with frame-level metadata and annotations.

## Top-level JSON Structure

Each video label file is a JSON object with these keys:

- `images`: frame-level metadata list
- `annotations`: action annotations per frame (can be multiple per frame)
- `categories`: taxonomy dictionary (`phase`, `instrument`, `verb`, `target`, `triplet`)
- `metadata`: file-level metadata/version info

From `C1V1.json`:

- `len(images) = 5332`
- `len(annotations) = 11679`
- category counts:
  - `phase: 12`
  - `instrument: 12`
  - `verb: 13`
  - `target: 15`
  - `triplet: 85`

## `images` Entry

Each item in `images` looks like:

```json
{
  "id": 0,
  "raw_id": 2850,
  "file_name": "",
  "width": 224,
  "height": 224,
  "fps": 25,
  "phase_label": 0,
  "phase_label_name": "preparation"
}
```

Important fields:

- `id`: internal frame id used for joining with `annotations.image_id`
- `raw_id`: original frame id from source timeline
- `file_name`: filename (the current loader uses `id` to build paths)
- `phase_label` / `phase_label_name`: surgical phase information
- Some frames have no phase annotations, in that case `phase_label` is set to `null`.
## `annotations` Entry

Each item in `annotations` looks like:

```json
{
  "image_id": 0,
  "raw_id": 2850,
  "category_id": 11,
  "instrument_id": 0,
  "verb_id": 1,
  "target_id": 4,
  "classnames": {
    "triplet": "grasper,retract,liver",
    "instrument": "grasper",
    "verb": "retract",
    "target": "liver"
  },
  "iscrowd": 0
}
```

Important fields:

- `image_id`: links this annotation to `images[id]`
- `category_id`: triplet class id (used for `label_ivt`)
- `instrument_id`: instrument class id (used for `label_i`)
- `verb_id`: verb class id (used for `label_v`)
- `target_id`: target class id (used for `label_t`)
- `classnames`: readable labels for inspection/debug

Notes:

- A single frame can have multiple annotations (multi-label case).
- The loader aggregates all annotations of the same frame and sets multi-hot labels.


## `categories` Schema

- `categories["phase"]`: list of `{id, name}`
- `categories["instrument"]`: list of `{id, name}`
- `categories["verb"]`: list of `{id, name}`
- `categories["target"]`: list of `{id, name}`
- `categories["triplet"]`: list of `{id, name}`

This mapping defines the canonical class-id-to-name lookup used by metrics and reporting.
