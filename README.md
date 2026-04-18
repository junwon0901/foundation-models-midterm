# Deep Learning Mid-term Assignment

Three Hugging Face vision foundation model demos: object detection, selected-target segmentation, and visual question answering.

## Setup

```bash
git clone https://github.com/junwon0901/foundation-models-midterm.git
cd foundation-models-midterm
```

Using Conda:

```bash
chmod +x setup.sh
./setup.sh
conda activate 2026010688
```

Using venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
chmod +x setup.sh
./setup.sh
```

If you want to force a runtime:

```bash
./setup.sh cpu
./setup.sh cu124
./setup.sh cu128
```

`setup.sh` reinstalls the PyTorch stack explicitly, then installs the pinned project dependencies. This avoids the common case where an older `torch` build stays in the environment and keeps forcing CPU fallback.

When `conda` is installed, `setup.sh` creates the environment automatically and installs everything into `2026010688` by default. You can override the env name:

```bash
./setup.sh cu124 myenv
conda activate myenv
```

## Run with helper script

```bash
chmod +x run.sh

./run.sh grounding-dino
./run.sh sam2
./run.sh qwen3-vl
```

The helper script is optional. The same demos can also be run directly:

```bash
python Grounding-DINO.py
python SAM2-base-plus.py
python Qwen3-VL-8B-Instruct.py
```

## Usage

### Grounding DINO

- Model: `IDEA-Research/grounding-dino-base`
- Task: open-vocabulary object detection
- Input: `samples/xai506_example_image.jpg`
- Prompt: `a glasses. a person. a chair. a table.`
- Output: `result/Grounding-DINO-base_result.png`

### SAM2

- Model: `facebook/sam2-hiera-base-plus`
- Task: selected-target segmentation
- Input: `samples/xai506_example_image.jpg`
- Left click: select one object/person to mask
- Right click: exclude another object/person or background
- Backspace: undo
- Enter: run segmentation
- Output: `result/SAM2-base-plus_result.png`

### Qwen3-VL

- Model: `Qwen/Qwen3-VL-2B-Instruct`
- Task: visual question answering
- Input: `samples/classroom_test.jpg`
- Ask multiple questions about the image in the terminal.
- Type `exit` or `quit` to stop.
- Use another image: `python Qwen3-VL-8B-Instruct.py --image samples/xai506_example_image.jpg`

Example conversation:

```text
$ python Qwen3-VL-8B-Instruct.py
Using device: mps
Image: samples/classroom_test.jpg
Ask questions about the image. Type 'exit' or 'quit' to stop.

You: How many people are in this image?
Qwen3-VL: There are several people sitting in a classroom.

You: What are they doing?
Qwen3-VL: They appear to be attending a class or lecture.

You: exit
```

## Results

### Grounding DINO Base

![Grounding DINO Base result](result/Grounding-DINO-base_result.png)

### SAM2 Hiera Base Plus

![SAM2 Hiera Base Plus result](result/SAM2-base-plus_result.png)

## Notes

- First run downloads model weights.
- Grounding DINO and SAM2 require a display for Matplotlib windows.
- Qwen3-VL can be heavy; CUDA is recommended.
- If `Using device: cpu` appears unexpectedly, rerun `./setup.sh cu124` and then check `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`.
