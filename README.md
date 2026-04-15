# Foundation Models Midterm

This repository contains three small demo scripts for testing vision foundation models:

- `Grounding-DINO.py`: zero-shot object detection with `IDEA-Research/grounding-dino-tiny`
- `SAM2-small.py`: interactive point-based segmentation with `facebook/sam2-hiera-small`
- `Qwen3-VL-#B-Instruct.py`: image question answering with `Qwen/Qwen3-VL-2B-Instruct`

The sample images are stored in `samples/`.

## Requirements

- Linux or macOS
- Python 3.10 or newer
- Internet connection for downloading models from Hugging Face
- A graphical display for the Matplotlib windows used by Grounding DINO and SAM2
- CUDA GPU recommended, especially for Qwen3-VL, but the scripts also try CPU when GPU is not available

## Setup

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
```

If you are using a CUDA-enabled Linux machine, install the PyTorch build that matches your CUDA version before installing the rest of the requirements. See the official PyTorch installation selector:

https://pytorch.org/get-started/locally/

Example:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirement.txt
```

## Run

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

SAM2 opens an image window. Left click adds a positive point, right click adds a negative point, Backspace removes the last point, and Enter runs segmentation.

Run Qwen3-VL:

```bash
./run.sh qwen3-vl
```

## Notes

The first run may take a long time because the model weights are downloaded from Hugging Face.

The Python files use fixed relative image paths. `run.sh` creates a local `foundation-models` symlink to `samples` when needed, so the scripts can run without changing their code.

If Matplotlib windows do not open on a remote Linux server, run the scripts from a machine with a display, enable X11 forwarding, or use a desktop environment.
