# Computer Vision Analysis Toolkit - Design Document

**Purpose:** This document enables anyone (including future AI assistants) to understand and extend the CV toolkit with zero prior context.

**Last Updated:** 2024  
**Version:** 2.0 (Modular Refactor)

---

## 🎯 System Overview

### What This Is

A modular Python framework for video analysis. Started as a water flow analyzer, evolved into a general-purpose CV toolkit supporting:

- Optical flow analysis
- Visual texture/light analysis
- Human pose estimation (body, hands, face)

### Core Philosophy

- **Modular:** Each analysis is independent, can run standalone
- **Extensible:** Adding new methods requires minimal changes to existing code
- **User-friendly:** Interactive menu + direct module execution
- **Reusable:** Shared preprocessing (stabilization + color enhancement)

---

## 📐 Architecture

### Directory Structure

```
cv_toolkit/
├── config.py                    # CENTRAL CONFIG - all parameters here
├── preprocessing.py             # Shared preprocessing pipeline
├── main.py                      # Menu orchestrator - CHANGES FREQUENTLY
├── modules/                     # Analysis methods organized by category
│   ├── __init__.py
│   ├── flow/                    # Motion & velocity analysis
│   │   ├── __init__.py
│   │   ├── lucas_kanade.py
│   │   ├── dis.py
│   │   └── velocity_magnitude.py
│   ├── visual/                  # Light & texture analysis
│   │   ├── __init__.py
│   │   ├── light_reflection.py
│   │   ├── light_rays.py
│   │   └── texture_analysis.py
│   └── pose/                    # Human pose estimation
│       ├── __init__.py
│       ├── human_pose.py
│       ├── hand_tracking.py
│       └── face_mesh.py
├── utils/                       # Shared utilities (future use)
│   └── __init__.py
└── output/                      # Generated videos
```

### Key Files to Understand for Iteration

**MUST READ for context:**

1. **This design doc** - Architecture and patterns
2. **main.py** - Menu system and orchestration logic
3. **config.py** - All configurable parameters
4. **README.md** - User-facing documentation

**Reference as needed:** 5. **preprocessing.py** - Preprocessing pipeline (rarely changes) 6. **Any module in modules/** - Example of analysis implementation

---

## 🔧 Core Components

### 1. config.py

**Purpose:** Single source of truth for all parameters

**Structure:**

```python
# Video settings
INPUT_VIDEO = 'video.mp4'
MAX_FRAMES = 300

# Preprocessing settings
STABILIZATION = {...}
COLOR_ENHANCEMENT = {...}

# Per-module settings
FLOW_LUCAS_KANADE = {...}
POSE_ESTIMATION = {...}
# etc.
```

**When to modify:** Adding new modules, tuning parameters

### 2. preprocessing.py

**Purpose:** Shared preprocessing pipeline

**Key Functions:**

- `load_video(path, max_frames)` → frames, metadata
- `stabilize_video(frames)` → stabilized_frames
- `enhance_colors(frame)` → enhanced_frame
- `preprocess_video(path, max_frames)` → frames, metadata (main entry point)

**Contract:** Returns `(frames, metadata)` where:

```python
frames = [numpy.ndarray, ...]  # List of BGR frames
metadata = {
    'fps': int,
    'width': int,
    'height': int,
    'total_frames': int
}
```

**When to modify:** Rarely. Only when adding new preprocessing steps.

### 3. main.py

**Purpose:** User interface and orchestration

**Key Functions:**

- `print_menu()` - Displays analysis options
- `get_analysis_categories()` - Returns dict of all analyses
- `run_analysis(choice, frames, metadata)` - Executes selected analysis
- `main()` - Main loop

**Structure:**

```python
categories = {
    'flow': {
        '1': ('Name', function_reference),
        ...
    },
    'visual': {...},
    'pose': {...}
}
```

**When to modify:** FREQUENTLY. Every time you add/remove analysis methods.

### 4. Analysis Modules (modules/_/_.py)

**Purpose:** Individual analysis implementations

**Required Contract:**
Every analysis module MUST implement:

```python
def analyze_method_name(frames, metadata):
    """
    Args:
        frames: List[np.ndarray] - BGR video frames
        metadata: Dict with keys: fps, width, height, total_frames

    Returns:
        str: Path to output video file
    """
    # 1. Setup output
    output_path = os.path.join(OUTPUT_DIR, 'output.mp4')

    # 2. Process frames
    for i, frame in enumerate(frames):
        # Analysis logic here
        pass

    # 3. Print statistics
    print(f"\nStatistics: ...")

    # 4. Return output path
    return output_path
```

**When to modify:** When adding new analysis methods or improving existing ones.

---

## 🔄 Data Flow

```
User Input
    ↓
main.py (menu selection)
    ↓
preprocessing.py (load & enhance video)
    ↓
frames + metadata
    ↓
modules/category/method.py (analysis)
    ↓
output/result.mp4
```

**Key Point:** Preprocessing runs ONCE, results are reused across multiple analyses in the same session.

---

## ➕ How to Add New Analysis Methods

### Step 1: Create the Module

Create `modules/category/new_method.py`:

```python
import cv2
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import OUTPUT_DIR, NEW_METHOD_CONFIG  # Add config if needed

def analyze_new_method(frames, metadata):
    """Your analysis here"""
    output_path = os.path.join(OUTPUT_DIR, 'new_method_output.mp4')

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, metadata['fps'],
                         (metadata['width'], metadata['height']))

    # Process frames
    for i, frame in enumerate(frames):
        result = frame.copy()

        # YOUR ANALYSIS LOGIC HERE

        out.write(result)

        if (i + 1) % 30 == 0:
            print(f"Processed {i + 1}/{len(frames)} frames")

    out.release()

    # Print statistics
    print(f"\nOutput saved to: {output_path}")

    return output_path

# Allow standalone execution
if __name__ == "__main__":
    from preprocessing import preprocess_video
    from config import INPUT_VIDEO, MAX_FRAMES

    frames, metadata = preprocess_video(INPUT_VIDEO, MAX_FRAMES)
    analyze_new_method(frames, metadata)
```

### Step 2: Add Configuration (if needed)

In `config.py`:

```python
NEW_METHOD_CONFIG = {
    'parameter1': value1,
    'parameter2': value2
}
```

### Step 3: Update main.py

**Location:** In `get_analysis_categories()` function

**Add to appropriate category:**

```python
def get_analysis_categories():
    return {
        'flow': {...},
        'visual': {...},
        'pose': {...},
        'new_category': {  # Or add to existing category
            'N': ('Method Name', analyze_new_method)
        }
    }
```

**Update imports at top:**

```python
from modules.new_category.new_method import analyze_new_method
```

**Update menu:**

```python
def print_menu():
    # Add new section or add to existing
    print("\n🆕 NEW CATEGORY")
    print("  N. New Method Name (description)")
```

**Update valid choices:**

```python
# In main() function
valid_choices = [str(i) for i in range(15)]  # Increment as needed
```

### Step 4: Test

```bash
# Test standalone
python modules/new_category/new_method.py

# Test via menu
python main.py
```

---

## 📦 Dependency Management

### Required Dependencies

```
opencv-python       # Core CV operations
numpy              # Numerical operations
matplotlib         # (Currently unused, legacy)
```

### Optional Dependencies

```
mediapipe          # Required ONLY for pose estimation modules
                   # Can skip if not using pose/hand/face tracking
```

### Installation Commands

```bash
# Minimum
pip install opencv-python numpy

# Full (with pose estimation)
pip install opencv-python numpy mediapipe
```

**Graceful Degradation:** Pose modules check for MediaPipe and display helpful error if missing.

---

## 🎨 Design Patterns

### 1. Category-Based Organization

Analysis methods grouped by purpose:

- **flow/** - Motion/velocity (any video with movement)
- **visual/** - Appearance/texture (light, patterns, turbulence)
- **pose/** - Human analysis (requires people in frame)

### 2. Configuration Injection

All parameters in `config.py`, imported by modules. Never hardcode values.

### 3. Standalone + Orchestrated Execution

Every module can run:

- Via `main.py` menu
- Directly: `python modules/category/method.py`

### 4. Consistent I/O Contract

- **Input:** `(frames, metadata)`
- **Output:** `output_path` string
- **Side effects:** Prints progress and statistics

### 5. Preprocessing Reuse

Preprocessing is expensive. Run once, reuse results for multiple analyses in same session.

---

## 🚧 Common Modifications

### Adding a New Category

1. Create `modules/new_category/` directory
2. Add `__init__.py`
3. Create analysis modules
4. Update `main.py`:
   - Import new modules
   - Add to `get_analysis_categories()`
   - Add menu section in `print_menu()`
   - Add batch option (optional)

### Adding Batch Operations

In `run_analysis()` function in `main.py`:

```python
elif choice == 'N':  # New batch number
    print("\nRunning ALL Category analyses...")
    outputs = []
    for name, func in categories['category_name'].values():
        print(f"\nStarting {name}...")
        output = func(frames, metadata)
        outputs.append(output)
        print(f"✓ {name} complete!")
    return outputs
```

### Changing Preprocessing Behavior

Modify `preprocessing.py` functions. Be careful - this affects ALL analyses.

**Toggle preprocessing:**

```python
# In config.py
STABILIZATION['enabled'] = False  # Disable stabilization
COLOR_ENHANCEMENT['enabled'] = False  # Disable color enhancement
```

### Changing Output Format

Currently hardcoded to MP4. To change:

In any module, modify:

```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec
output_path = os.path.join(OUTPUT_DIR, 'output.avi')  # Change extension
```

---

## 🐛 Debugging Tips

### Module Not Found Errors

**Cause:** Python path issues when running modules standalone

**Fix:** Each module includes:

```python
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
```

This adds project root to path for imports.

### MediaPipe Errors

**Cause:** MediaPipe not installed

**Fix:**

```bash
pip install mediapipe
```

Or disable pose modules in menu if not needed.

### Memory Issues

**Symptoms:** Crashes, slow performance

**Solutions:**

- Reduce `MAX_FRAMES` in config.py
- Process video in chunks
- Lower pose model complexity: `POSE_ESTIMATION['model_complexity'] = 0`

### Slow Processing

**Optimization options:**

- Disable stabilization: `STABILIZATION['enabled'] = False`
- Lower DIS preset: `FLOW_DIS['preset'] = 0` (fast mode)
- Skip preprocessing: Answer 'n' when prompted

---

## 📊 Testing New Modules

### Checklist

- [ ] Module runs standalone: `python modules/category/method.py`
- [ ] Module accessible via menu
- [ ] Output video generated in `output/`
- [ ] Statistics printed to console
- [ ] Progress updates every 30 frames
- [ ] Returns output path string
- [ ] Handles edge cases (no detection, empty frames)

### Test Videos

Recommended test videos for each category:

- **flow/**: Any video with motion (water, traffic, sports)
- **visual/**: Videos with light/texture (water, sunsets, sparkles)
- **pose/**: Videos with people (talking heads, sports, dance)

---

## 🔮 Future Expansion Ideas

### Potential New Categories

```
modules/
├── tracking/          # Object tracking (CSRT, KCF, SORT)
├── segmentation/      # Semantic segmentation, background subtraction
├── detection/         # Object detection (YOLO, etc.)
├── depth/             # Depth estimation, stereo vision
└── analytics/         # Statistical analysis, plotting, exports
```

### Potential Features

- Export statistics to CSV/JSON
- Real-time webcam analysis
- Batch process multiple videos
- GPU acceleration support
- Web interface (Flask/Streamlit)
- Comparison tools (side-by-side analysis)

---

## 📝 Version History

### v2.0 - Modular Refactor (Current)

- Reorganized into category-based modules
- Added pose estimation (human, hand, face)
- Improved menu system with batch operations
- Generalized from water-only to any video

### v1.0 - Initial Water Analysis

- Optical flow (Lucas-Kanade, DIS, Farneback)
- Light reflection analysis
- Light ray projection
- Texture/turbulence analysis
- Velocity magnitude heatmaps

---

## 🤝 Contributing Guidelines

### Code Style

- Use descriptive variable names
- Add docstrings to functions
- Print progress every 30 frames
- Follow existing module structure

### Naming Conventions

- Files: `snake_case.py`
- Functions: `snake_case()`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`

### Git Workflow (if versioning)

```bash
# Feature branch
git checkout -b feature/new-analysis-method

# Commit changes
git commit -m "Add: New analysis method for X"

# Merge back
git checkout main
git merge feature/new-analysis-method
```

---

## 🆘 Getting Help

### For AI Assistants

When helping someone iterate on this project:

1. **Always read:**

   - This design doc (context)
   - main.py (current menu structure)
   - README.md (user documentation)

2. **Ask clarifying questions:**

   - What category does this belong to?
   - What input/output format?
   - Any special dependencies?

3. **Follow patterns:**
   - Match existing module structure
   - Use consistent naming
   - Add to config.py if parameters needed

### For Humans

**Quick start:** Read README.md first

**Adding features:** Read this design doc

**Debugging:** Check "Debugging Tips" section

**Questions:**

- Check if similar module exists
- Review existing module implementations
- Consult OpenCV/MediaPipe documentation

---

## 📚 Key Concepts

### Optical Flow

Tracks motion between frames. Two types:

- **Sparse (Lucas-Kanade):** Tracks specific feature points
- **Dense (DIS, Farneback):** Tracks every pixel

**Use cases:** Water flow, traffic, sports, any motion

### Texture Analysis

Quantifies visual patterns using:

- Gradients (edge detection)
- Gabor filters (oriented patterns)
- Variance (local chaos/turbulence)

**Use cases:** Water turbulence, surface patterns, material classification

### Pose Estimation

Detects and tracks human body landmarks using MediaPipe.

**Models:**

- **Pose:** 33 body landmarks (full body)
- **Hands:** 21 landmarks per hand
- **Face:** 468-478 facial landmarks

**Use cases:** Activity recognition, gesture control, sports analysis

---

## 🎓 Learning Resources

### OpenCV

- Official docs: https://docs.opencv.org/
- Optical flow tutorial: Search "OpenCV optical flow"

### MediaPipe

- Official docs: https://google.github.io/mediapipe/
- Pose guide: MediaPipe Pose
- Hands guide: MediaPipe Hands

### Computer Vision Concepts

- Optical flow: Wikipedia "Optical flow"
- Gabor filters: Search "Gabor filter tutorial"
- Pose estimation: Search "pose estimation explained"

---

## ✅ Quick Reference

### Adding a New Analysis Method

1. Create `modules/category/method.py` with `analyze_method_name()` function
2. Add config to `config.py` (if needed)
3. Import in `main.py`
4. Add to `get_analysis_categories()` in `main.py`
5. Update menu in `print_menu()` in `main.py`
6. Test standalone and via menu

### File Modification Frequency

- **main.py:** Changes frequently (every new method)
- **config.py:** Changes frequently (new parameters)
- **modules/\*:** Changes when adding/modifying methods
- **preprocessing.py:** Rarely changes
- **README.md:** Update when adding major features
- **This doc:** Update when architecture changes

### Common Import Pattern

```python
import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import OUTPUT_DIR, METHOD_CONFIG
```

### Common Video Writer Setup

```python
output_path = os.path.join(OUTPUT_DIR, 'method_output.mp4')
os.makedirs(OUTPUT_DIR, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, metadata['fps'],
                     (metadata['width'], metadata['height']))

# ... process frames ...

out.release()
return output_path
```

---

**End of Design Document**

_This document should provide complete context for future iteration. If anything is unclear, update this document!_
