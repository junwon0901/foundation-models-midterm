from transformers import Sam2Processor, Sam2Model
import torch
from PIL import Image
import matplotlib.pyplot as plt

model_id = "facebook/sam2-hiera-small"
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

def get_click_inputs(image):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image)
    ax.set_title(
        "Left click = positive, Right click = negative, Enter = finish, Backspace = undo"
    )
    ax.axis("on")

    points = []
    labels = []
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
        points.append([x, y])
        labels.append(label)

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
            if points:
                removed_point = points.pop()
                removed_label = labels.pop()
                marker_artists.pop().remove()
                click_type = "positive" if removed_label == 1 else "negative"
                print(f"Removed last {click_type} click at ({removed_point[0]}, {removed_point[1]})")
                redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if not points:
        raise RuntimeError("No points were clicked.")

    input_points = [[points]]
    input_labels = [[labels]]
    return input_points, input_labels

input_points, input_labels = get_click_inputs(raw_image)
print(f"Input points: {input_points}")
print(f"Input labels: {input_labels}")

inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

# The model outputs multiple mask predictions ranked by quality score
print(f"Generated {masks.shape[1]} masks with shape {masks.shape}")

best_mask = masks[0, 0].float().numpy()

plt.figure(figsize=(10, 7))
plt.imshow(raw_image)
plt.imshow(best_mask, alpha=0.5)
plt.title("Best predicted mask")
plt.axis("off")
plt.show()
