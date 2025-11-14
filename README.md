# ImDim Image & Dimension Based CAD Modeling

## Problem description / motivation

The goal of this project is to input images or sets of dimensions and their relative orientations, and
generate a 3D model of a part. Further, we can use natural language or hand sketches to create a baseline
or target shape, and iteratively update the model until the desired details are achieved.

Inputs may range from precise measurements (e.g., calipers) to images taken by the user. We can also
attempt to invert this process or build a new functional part around an existing model using the same pipeline.

---

## Approach

### User Experience
- User inputs images and optionally measurements of the target item.
- User selects primitives and simple operations available to the synthesizer from a larger set (e.g., OpenSCAD).
- User iteratively adds more measurements, images, and other specifications.

### Method
- **CNN** to determine which primitive objects to include in the search space — an “automatic sketching” step.
  - Generate synthetic training data for the CNN.
- Explore multiple synthesis or hybrid strategies to optimally fill the shape using primitives and transforms:
  - Simple enumeration
  - Divide-and-Conquer synthesis  
    - Break the outlines into manageable segments.
  - **CEGIS**
    - Guide enumeration using feedback on regions not yet filled.
  - Other possible hybrid strategies?

### Control Flow
- Accept images and optionally measurements of target item.
- Use image processing to edge-detect and draw dimensions on the 2D outline.
- CNN produces an initial sketch for the synthesizer to build upon.
- Synthesis method generates candidate 3D models.
- Solver scores models based on how well they satisfy outlines, measurements, and other data.
- System prompts user to add more details (additional images, measurements, operations) to refine the model.

---

## Evaluation

### Research Questions
- Is this optimal packing / reconstruction approach comparable to existing methods (Szalinski, photogrammetry, LLM-based generation) within **5%, 10%, 20%** geometric error?
- After each iteration of user input, how much does the model improve?
  - What is the ideal number of images, measurement precision, and user steps versus overall user satisfaction?
- How effectively does a new image, measurement, or constraint prune the existing search space?

### Assessment Criteria
- Outline packing quality (% filled).
- As new dimensions and alternate-view outlines are added:
  - How well are new outlines packed?
  - How well is consistency maintained with previous outlines?
- Geometric accuracy (Chamfer distance, Hausdorff distance, F-score).
  - Especially using synthetic data when reconstructing known reference models.
