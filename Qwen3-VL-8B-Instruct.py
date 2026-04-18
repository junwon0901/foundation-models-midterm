import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model_id = "Qwen/Qwen3-VL-8B-Instruct"
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype="auto").to(device)
print(f"Using device: {device}")

processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "samples/xai506_example_image.jpg",
            },
            {"type": "text", "text": "How many people are in this image?. Answer in one sentence."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
