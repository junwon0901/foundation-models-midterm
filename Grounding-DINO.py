import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

model_id = "IDEA-Research/grounding-dino-tiny"
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image_path = "foundation-models/classroom_test.jpg"
image = Image.open(image_path)
# Check for cats and remote controls
# VERY important: text queries need to be lowercased + end with a dot
text = "a glasses. a person. a chair. a table."

inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

result = results[0]
boxes = result["boxes"].cpu()
scores = result["scores"].cpu()
labels = result["labels"]

fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image)

for box, score, label in zip(boxes, scores, labels):
    x_min, y_min, x_max, y_max = box.tolist()
    width = x_max - x_min
    height = y_max - y_min

    rect = patches.Rectangle(
        (x_min, y_min),
        width,
        height,
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )
    ax.add_patch(rect)
    ax.text(
        x_min,
        y_min - 5,
        f"{label}: {score:.2f}",
        fontsize=10,
        color="red",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
    )

ax.set_title("Grounding DINO detections")
ax.axis("off")
plt.show()