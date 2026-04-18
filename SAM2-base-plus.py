from transformers import Sam2Processor, Sam2Model
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

model_id = "facebook/sam2-hiera-base-plus"
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
model = Sam2Model.from_pretrained(model_id).to(device)
print(f"Using device: {device}")

processor = Sam2Processor.from_pretrained(model_id)

image_path = "samples/xai506_example_image.jpg"
raw_image = Image.open(image_path).convert("RGB")
result_dir = Path("result")
result_path = result_dir / "SAM2-base-plus_result.png"

def get_click_inputs(image):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image)
    ax.set_title(
        "Left click = target object/person, Right click = exclude, Enter = finish, Backspace = undo"
    )
    ax.axis("on")

    clicks = []
    marker_artists = []
    finished = {"done": False}

    def redraw():
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        if event.button == 1:
            label = 1
            marker_style = "+"
            marker_color = "lime"
        elif event.button == 3:
            label = 0
            marker_style = "x"
            marker_color = "red"
        else:
            return

        x, y = int(event.xdata), int(event.ydata)
        clicks.append(([x, y], label))

        artist, = ax.plot(
            x,
            y,
            marker=marker_style,
            color=marker_color,
            markersize=12,
            markeredgewidth=2,
        )
        marker_artists.append(artist)

        click_type = "positive" if label == 1 else "negative"
        print(f"Added {click_type} click at ({x}, {y})")
        redraw()

    def on_key(event):
        if event.key == "enter":
            finished["done"] = True
            plt.close(fig)
        elif event.key == "backspace":
            if clicks:
                removed_point, removed_label = clicks.pop()
                marker_artists.pop().remove()
                click_type = "positive" if removed_label == 1 else "negative"
                print(f"Removed last {click_type} click at ({removed_point[0]}, {removed_point[1]})")
                redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    positive_points = [point for point, label in clicks if label == 1]
    negative_points = [point for point, label in clicks if label == 0]

    if not positive_points:
        raise RuntimeError("No target objects or people were clicked.")

    # Treat each positive click as a separate target. Negative clicks are shared
    # across all targets to suppress unselected objects/people or background.
    object_points = []
    object_labels = []
    for positive_point in positive_points:
        object_points.append([positive_point] + negative_points)
        object_labels.append([1] + [0] * len(negative_points))

    input_points = [object_points]
    input_labels = [object_labels]
    return input_points, input_labels, clicks

input_points, input_labels, clicks = get_click_inputs(raw_image)
print(f"Input points: {input_points}")
print(f"Input labels: {input_labels}")

inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

# The model outputs multiple mask predictions per selected object.
print(f"Generated masks with shape {masks.shape}")

best_masks = masks[:, 0].float().numpy()
combined_mask = np.any(best_masks > 0, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].imshow(raw_image)
axes[0].set_title("Clicked points")
axes[0].axis("off")

for point, label in clicks:
    x, y = point
    if label == 1:
        marker_style = "+"
        marker_color = "lime"
    else:
        marker_style = "x"
        marker_color = "red"

    axes[0].plot(
        x,
        y,
        marker=marker_style,
        color=marker_color,
        markersize=12,
        markeredgewidth=2,
    )

axes[1].imshow(raw_image)
axes[1].imshow(combined_mask, alpha=0.5)
axes[1].set_title("Selected target masks")
axes[1].axis("off")

fig.tight_layout()
result_dir.mkdir(exist_ok=True)
fig.savefig(result_path, dpi=200, bbox_inches="tight")
print(f"Saved result to {result_path}")
plt.show()
