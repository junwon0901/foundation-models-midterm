# Deep Learning Mid-term Assignment

This repository contains a hands-on comparison of three vision foundation models from Hugging Face. Each script applies one model to a different computer vision task using sample images, so the models can be tested independently and their behavior can be compared in practice.

## Overview

The project focuses on three representative multimodal or vision foundation model tasks:

| Script | Model | Task | Hugging Face ID |
|--------|-------|------|-----------------|
| `Grounding-DINO.py` | Grounding DINO Base | Open-vocabulary object detection | `IDEA-Research/grounding-dino-base` |
| `SAM2-base-plus.py` | SAM2 Hiera Base Plus | Interactive point-based segmentation | `facebook/sam2-hiera-base-plus` |
| `Qwen3-VL-8B-Instruct.py` | Qwen3-VL 8B Instruct | Visual question answering | `Qwen/Qwen3-VL-8B-Instruct` |

The sample images are stored in `samples/`, and each script reads images from that directory relative to the repository root.

## Project Structure

```text
foundation-models/
|-- Grounding-DINO.py            # Open-vocabulary object detection demo
|-- SAM2-base-plus.py            # Interactive SAM2 segmentation demo
|-- Qwen3-VL-8B-Instruct.py      # Visual question answering demo
|-- samples/
|   |-- classroom_test.jpg       # Input image for Grounding DINO and SAM2
|   `-- xai506_example_image.jpg # Input image for Qwen3-VL
|-- result/
|   |-- Grounding-DINO-base_result.png
|   `-- SAM2-base-plus_result.png
|-- run.sh                       # Unified command-line runner
|-- requirement.txt              # Python dependencies
`-- README.md
```

## Model Details

### 1. Open-Vocabulary Object Detection - Grounding DINO Base

Grounding DINO detects objects in an image from a natural-language text prompt. Unlike a closed-set detector, it is not limited to a fixed list of predefined categories.

In `Grounding-DINO.py`, the prompt is defined as:

```python
text = "a glasses. a person. a chair. a table."
```

Important prompt format:

- Use lowercase object names.
- Separate object descriptions with periods.
- End the prompt with a period.

Output behavior:

- Loads `samples/classroom_test.jpg`.
- Runs zero-shot object detection.
- Draws bounding boxes, labels, and confidence scores.
- Saves the result to `result/Grounding-DINO-base_result.png`.
- Displays the result in a Matplotlib window.

Core flow:

```text
Load processor and model
-> Load input image
-> Encode image and text prompt
-> Run inference with torch.no_grad()
-> Post-process boxes, labels, and scores
-> Draw detections with Matplotlib
-> Display result
```

### 2. Interactive Selected-Student Segmentation - SAM2 Hiera Base Plus

SAM2 segments selected students from user-provided point prompts. Each left click is treated as a separate selected student, and right clicks are used as shared negative prompts to suppress unselected students or background regions.

Interaction controls:

| Action | Meaning |
|--------|---------|
| Left click | Select one student to mask |
| Right click | Mark a student or background region to exclude |
| Backspace | Remove the most recent point |
| Enter | Finish point selection and run segmentation |

Output behavior:

- Loads `samples/classroom_test.jpg`.
- Opens an interactive Matplotlib window.
- Collects positive and negative point prompts.
- Runs SAM2 segmentation for each selected student.
- Saves a side-by-side click and mask result to `result/SAM2-base-plus_result.png`.
- Displays the same side-by-side result in a Matplotlib window.

Core flow:

```text
Load processor and model
-> Load input image
-> Open interactive click window
-> Collect point coordinates and labels
-> Encode each positive click as a separate target object
-> Run inference with torch.no_grad()
-> Post-process predicted masks
-> Combine selected-student masks
-> Save and display a side-by-side click/mask result
```

### 3. Visual Question Answering - Qwen3-VL-8B-Instruct

Qwen3-VL is a vision-language model that answers natural-language questions about an image. This demo sends one image and one text question to the model, then prints the generated answer.

The default question in `Qwen3-VL-8B-Instruct.py` is:

```text
How many people are in this image?. Answer in one sentence.
```

Output behavior:

- Loads `samples/xai506_example_image.jpg`.
- Applies the Qwen chat template.
- Generates an answer with `max_new_tokens=128`.
- Prints the decoded answer to the terminal.

Core flow:

```text
Load processor and model
-> Prepare image-text message
-> Apply chat template
-> Encode multimodal input
-> Generate answer
-> Trim prompt tokens from generated output
-> Decode and print answer
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/junwon0901/foundation-models-midterm.git
cd foundation-models-midterm
```

### 2. Create a Python Environment

Using `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Using Conda:

```bash
conda create -n 2026010688 python=3.10 -y
conda activate 2026010688
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
```

For CUDA-enabled Linux machines, install the PyTorch build that matches your CUDA version before installing the remaining dependencies. See the official PyTorch selector:

https://pytorch.org/get-started/locally/

Example for CUDA 12.1:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirement.txt
```

## How to Run

Make the runner executable:

```bash
chmod +x run.sh
```

List available demos:

```bash
./run.sh list
```

Run Grounding DINO:

```bash
./run.sh grounding-dino
```

Run SAM2:

```bash
./run.sh sam2
```

Run Qwen3-VL:

```bash
./run.sh qwen3-vl
```

The scripts can also be run directly:

```bash
python Grounding-DINO.py
python SAM2-base-plus.py
python Qwen3-VL-8B-Instruct.py
```

Run commands from the repository root so the scripts can find the files inside `samples/`.

## Output Summary

| Demo | Output |
|------|--------|
| Grounding DINO | Matplotlib window and saved image with bounding boxes, object labels, and confidence scores |
| SAM2 | Interactive point-selection window, followed by a saved side-by-side click and mask result |
| Qwen3-VL | Text answer printed in the terminal |

Grounding DINO and SAM2 save visual outputs in the `result/` directory.

### Saved Result Images

#### Grounding DINO Base

![Grounding DINO Base result](result/Grounding-DINO-base_result.png)

#### SAM2 Hiera Base Plus

![SAM2 Hiera Base Plus result](result/SAM2-base-plus_result.png)

## Environment

| Item | Requirement |
|------|-------------|
| OS | Linux or macOS recommended |
| Python | 3.10 or newer |
| Device | CUDA, Apple MPS, or CPU |
| Internet | Required on first run to download model weights |
| Hugging Face token | Not required for the listed public models |
| Display | Required for Matplotlib windows in Grounding DINO and SAM2 |

Qwen3-VL-8B can be slow on CPU and needs substantially more memory than the 2B variant. A CUDA GPU is recommended for faster inference, especially for repeated testing.

## Notes

- The first run may take several minutes because model weights are downloaded from Hugging Face.
- If Matplotlib windows do not open on a remote Linux server, run the scripts on a machine with a display, enable X11 forwarding, or use a desktop environment.
- If you change the input image paths inside the Python files, keep the paths consistent with the repository structure.
- See `requirement.txt` for the full dependency list.
