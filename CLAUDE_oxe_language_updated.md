# Language Action Conditioning for Cosmos-Predict2.5 Video Model

## Project Overview

This project implements **language action conditioning** for video prediction using
Cosmos-Predict2.5. Instead of conditioning the video model on numerical 7-dim delta
EEF actions, we replace the numerical action input with natural language descriptions
of per-frame robot motions. The task instruction is kept as a separate input — language
actions do not replace or merge with the task prompt, they replace the numerical action
input.

---

## Core Concept

Standard action-conditioned video prediction:
```
[task text prompt] + [7-dim numerical delta action] → predicted video frames
```

This project's approach:
```
[task text prompt] + [language action chunk] → predicted video frames
```

The two text inputs serve different roles:
- **Task prompt** — describes WHAT the robot is trying to accomplish (goal-level)
- **Language action** — describes HOW the robot moves at each frame (motion-level)

This separation enables cross-embodiment generalization: language actions like
"move forward, open gripper" are robot-agnostic and require no action space
normalization across different robot embodiments.

---

## Dataset

### Datasets Used in This Experiment

All datasets are already converted to MP4 and stored on the cluster. Language
conversion is already completed — per-frame language labels and task instructions
are available under each dataset's `language/` folder.

| Dataset          | Robot         | Path                                                        |
|------------------|---------------|-------------------------------------------------------------|
| Bridge V2        | WidowX 250    | `/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4/bridge`  |
| DROID            | Franka Panda  | `/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4/droid`   |
| Fractal (RT-1)   | Google Robot  | `/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4/fractal` |
| Taco Play        | Franka Panda  | `/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4/taco_play`|
| BC-Z             | Google Robot  | `/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4/bc_z`    |
| FMB              | Franka Panda  | `/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4/fmb`     |
| Furniture Bench  | Franka Panda  | `/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4/furniture_bench` |

### Dataset Directory Structure

Each dataset folder follows this structure:

```
/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4/<dataset_name>/
├── language/                      ← ALREADY COMPLETE — do not regenerate
│   ├── <episode_id>.json          ← per-episode file containing:
│   │                                 - task_description (episode-level instruction)
│   │                                 - frames[]: list of per-frame action language
│   │                                   e.g. {"frame_idx": 0, "actions": "move forward, move left"}
│   └── ...
└── videos/
    ├── train/
    │   ├── <episode_id>/
    │   │   └── rgb.mp4            ← main camera view
    │   └── ...
    └── test/
```

### Language Folder Contents

The `language/` folder under each dataset contains everything needed for
conditioning — no further action-to-language conversion is required:

- **Task description**: episode-level natural language goal instruction
- **Per-frame action language**: one label per video frame describing robot motion
  at that timestep (e.g. `"move forward, move left, open gripper"` or
  `"no significant motion"`)

To load language for an episode:
```python
import json, os

def load_language(dataset_root, episode_id):
    lang_path = os.path.join(dataset_root, 'language', f'{episode_id}.json')
    with open(lang_path) as f:
        data = json.load(f)
    task        = data['task_description']
    frame_langs = [fr['actions'] for fr in data['frames']]
    return task, frame_langs   # str, list[str]
```

---

## Action Space (Reference Only)

The action-to-language conversion is already complete. The following documents
the 7-dim delta EEF format used across datasets for reference and for the
numerical action baseline (Experiment 1).

Target format: `[dx, dy, dz, droll, dpitch, dyaw, gripper]` (7-dim)

All deltas were computed from consecutive absolute EEF states:
```python
# Bridge V2 example (absolute state → delta)
xyz_delta = state[t+1, :3] - state[t, :3]
prev_rotm = Rotation.from_euler('xyz', state[t, 3:6]).as_matrix()
curr_rotm = Rotation.from_euler('xyz', state[t+1, 3:6]).as_matrix()
rel_trans = prev_rotm.T @ xyz_delta           # delta in previous EEF frame
rel_rot   = Rotation.from_matrix(prev_rotm.T @ curr_rotm).as_euler('xyz')
gripper   = state[t+1, 6]                     # taken as-is from current frame
```

### BC-Z Note
BC-Z stores rotation as axis-angle. It was converted to Euler before delta computation:
```python
from scipy.spatial.transform import Rotation
aa  = np.array(ctx['steps/observation/present/axis_angle']).reshape(T, 3)
rpy = Rotation.from_rotvec(aa).as_euler('xyz')   # (T, 3) → [roll, pitch, yaw]
xyz = np.array(ctx['steps/observation/present/xyz']).reshape(T, 3)
arm_states = np.concatenate([xyz, rpy], axis=1)  # (T, 6)
```

---

## Language Action Chunk Format

### Chunk Size
```
K = 12 frames per chunk
```

### Windowing Strategy
Non-overlapping chunks (simpler, sufficient for first experiment):
```
chunk 1: frames [0  - 11 ]
chunk 2: frames [12 - 23 ]
chunk 3: frames [24 - 35 ]
...
```
Last chunk is zero-padded with "no significant motion" if episode length not divisible by 12.

### Text Format
Two independent strings encoded separately by T5:

**Stream 1 — task text** (episode-level, one per episode):
```
move the spatula to the far left corner behind the pot
```

**Stream 2 — action chunk text** (chunk-level, one per K=12 frames):
```
step1: move forward, move left, move up, tilt backward.
step2: move forward, move left, move up, tilt backward, rotate right.
step3: move forward, move left, move up, tilt backward, rotate right.
step4: move up, roll counter-clockwise, tilt backward.
step5: move backward, move up, roll counter-clockwise, tilt backward.
step6: move backward, move right, move up, roll counter-clockwise, tilt backward.
step7: no significant motion.
step8: move backward, move right, move up, roll counter-clockwise, rotate left.
step9: move backward, move right, move up, roll counter-clockwise, tilt backward, rotate left.
step10: move backward, move right, move up, tilt backward, rotate left.
step11: move right, roll clockwise.
step12: no significant motion.
```

### Token Budget
```
Task instruction:      ~15 tokens
Separators:            ~5  tokens
Per step (avg):        ~10 tokens
12 steps × 10 tokens:  ~120 tokens
Total per chunk:       ~140 tokens
T5 encoder limit:       512 tokens  ✅ well within budget
```

### Formatting Function
```python
def format_chunk(task_description, action_labels, start_frame, K=12):
    """
    Returns two independent strings to be encoded separately by T5.

    Args:
        task_description: str — episode-level task instruction
        action_labels:    list[str] — per-frame language action labels for full episode
        start_frame:      int — first frame index of this chunk
        K:                int — chunk size (default 12)

    Returns:
        task_text:   str — encode independently as Stream 1
        action_text: str — encode independently as Stream 2
    """
    steps = []
    for i in range(K):
        idx = start_frame + i
        if idx < len(action_labels):
            label = action_labels[idx]
        else:
            label = 'no significant motion'   # pad last chunk
        steps.append(f"step{i+1}: {label}.")

    task_text   = task_description
    action_text = ' '.join(steps)
    return task_text, action_text  # two separate strings — encode independently
```

Example outputs for one chunk:
```python
task_text   = "move the spatula to the far left corner behind the pot"

action_text = "step1: move forward, move left, move up, tilt backward. \
               step2: move forward, move left, move up, tilt backward, rotate right. \
               step3: open gripper. \
               step4: no significant motion. \
               step5: move backward, tilt forward. \
               step6: move backward, move left, move down. \
               step7: move backward, move left, move down, rotate left. \
               step8: close gripper. \
               step9: move forward, move left, close gripper. \
               step10: no significant motion. \
               step11: move forward, roll clockwise, tilt backward. \
               step12: no significant motion."

# Encode separately:
task_embedding   = T5(task_text)    # Stream 1 — goal-level
action_embedding = T5(action_text)  # Stream 2 — motion-level
```

---

## Experiment

### Language Action Conditioning
```
Input:   [task text embedding] + [language action chunk embedding] (two independent streams)
Output:  next 12 video frames
Datasets: bc_z, bridge, droid, fmb, fractal, furniture_bench, taco_play
Purpose: Train Cosmos-Predict2.5 conditioned on language actions as a drop-in
         replacement for numerical actions
```

### Two Independent Text Embeddings

Task text and language action chunk are encoded **separately** by the T5 text encoder
and injected into Cosmos as two independent conditioning streams — they are never
concatenated into a single string.

```
Stream 1 — Task embedding:
    input:  "move the spatula to the far left corner behind the pot"
    encode: T5(task_text) → task_embedding
    role:   WHAT the robot is trying to accomplish (goal-level, episode-level)

Stream 2 — Action chunk embedding:
    input:  "step1: move forward, move left. step2: open gripper. ... step12: ..."
    encode: T5(action_chunk_text) → action_embedding
    role:   HOW the robot moves at each frame (motion-level, chunk-level)
```

These two embeddings are kept independent throughout the model — do not merge,
concatenate, or combine them into a single string before encoding.

---

## Data Pipeline

### Status
- ✅ MP4 conversion: complete for all 7 datasets
- ✅ Action-to-language conversion: complete — language files are in `language/` folders
- 🔲 Chunk building: next step — read language files, build K=12 conditioning strings
- 🔲 Training: Cosmos post-training with language action chunks

### Next Step — Build K=12 Chunks
Read the pre-computed language from each dataset's `language/` folder and
format into K=12 conditioning strings for Cosmos training:

```python
BASE = "/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4"
DATASETS = ["bc_z", "bridge", "droid", "fmb", "fractal", "furniture_bench", "taco_play"]

for dataset in DATASETS:
    lang_dir   = os.path.join(BASE, dataset, "language")
    # iterate over episode JSON files in lang_dir
    # load task + per-frame labels
    # call format_chunk() for each non-overlapping window of K=12
    # write chunks to training manifest
```

### Training Manifest Format
One JSON line per chunk (JSON Lines format for efficient streaming):
```json
{
    "dataset": "bridge",
    "episode_id": "0",
    "start_frame": 0,
    "end_frame": 11,
    "video_path": "/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4/bridge/videos/train/0/rgb.mp4",
    "task_text": "move the spatula to the far left corner behind the pot",
    "action_text": "step1: move forward, move left. step2: open gripper. step3: ..."
}
```

Note: `task_text` and `action_text` are stored as two separate fields and must be
encoded independently by T5 at training time — never concatenate them.

---

## Critical Implementation Notes

### Language Files Are Already Complete
Do NOT re-run action-to-language conversion. Read directly from:
```
/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4/<dataset>/language/<episode_id>.json
```

### Frame Alignment
Language action at frame `t` describes the motion from frame `t` to frame `t+1`.
The chunk for frames `[t, t+K)` is paired with the video clip for those same frames.
Off-by-one errors silently corrupt the action-video correspondence — verify carefully.

### Gripper Events
Gripper transitions (open/close) are the highest-information frames in any manipulation
episode. Verify that gripper events appear correctly in the language chunks by spot-checking
a few episodes from each dataset before training.

### Empty Language Instructions
DROID has some episodes with empty task descriptions (`"texts": [""]`).
Filter these out for Cosmos training or treat as unconditional examples
(empty string for task prompt, keep language action chunk).

### Chunk Padding
If episode length T is not divisible by K=12, the last chunk is padded:
```python
label = frame_langs[idx] if idx < len(frame_langs) else 'no significant motion'
```

---

## File Structure

```
/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4/
├── bc_z/
│   ├── language/          ✅ done — per-episode JSON with task + per-frame labels
│   └── videos/
├── bridge/
│   ├── language/          ✅ done
│   └── videos/
├── droid/
│   ├── language/          ✅ done
│   └── videos/
├── fmb/
│   ├── language/          ✅ done
│   └── videos/
├── fractal/
│   ├── language/          ✅ done
│   └── videos/
├── furniture_bench/
│   ├── language/          ✅ done
│   └── videos/
└── taco_play/
    ├── language/          ✅ done
    └── videos/
```

---

## Cluster Setup (Princeton Della-GPU)

```bash
# Account and user
account: seas / AM43
user:    kl0820

# All dataset base path (MP4 + language — all ready)
/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4/

# Datasets available
bc_z  bridge  droid  fmb  fractal  furniture_bench  taco_play

# Language files location (per dataset)
/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4/<dataset>/language/
```

---

## Dependencies

```bash
pip install tensorflow tensorflow-datasets scipy opencv-python
pip install torch torchvision
pip install transformers  # T5 tokenizer for token budget checks
```
