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
- Tested with: 1 x NVIDIA GeForce RTX 4090 24GB
- Ask multiple questions about the image in the terminal.
- Type `exit` or `quit` to stop.
- Use another image: `python Qwen3-VL-8B-Instruct.py --image samples/xai506_example_image.jpg`

Example conversation:

```text
$ python Qwen3-VL-8B-Instruct.py
Using device: cuda
Image: samples/xai506_example_image.jpg
Ask questions about the image. Type 'exit' or 'quit' to stop.

You: How many people are in this image?
Qwen3-VL: Based on a careful count of the individuals visible in the image, there are **22 people** in the classroom.

Here is a breakdown of the count:

- **Front row (closest to the camera):** 3 people (one on the far left, one in the center, and one on the right).
- **Second row:** 4 people.
- **Third row:** 5 people.
- **Fourth row:** 5 people.
- **Fifth row (back row):** 5 people.

This totals to 3 + 4 + 5 + 5 + 5 = 22 people.

You: What are they doing?
Qwen3-VL: Based on the image, the people appear to be attending a **class or lecture**.

Here's a more detailed description of what they are doing:

- **Most are focused on their laptops:** Many students have open laptops on their desks, suggesting they are taking notes, following along with a presentation, or working on assignments related to the class.
- **One person is actively speaking:** The man in the light grey t-shirt in the middle-right of the room is gesturing with his hands and looking forward, likely participating in a discussion or asking a question.
- **One person appears to be resting:** The man on the far left is

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
