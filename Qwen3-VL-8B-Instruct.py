import argparse
from pathlib import Path

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_IMAGE = "samples/classroom_test.jpg"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Chat with Qwen3-VL about one image across multiple turns."
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=f"Image path to discuss. Default: {DEFAULT_IMAGE}",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate per answer.",
    )
    return parser.parse_args()


def make_user_message(question, image_path=None):
    content = []
    if image_path is not None:
        content.append({"type": "image", "image": str(image_path)})
    content.append({"type": "text", "text": question})
    return {"role": "user", "content": content}

def make_assistant_message(answer):
    return {"role": "assistant", "content": [{"type": "text", "text": answer}]}

def generate_answer(model, processor, messages, max_new_tokens):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()

def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = get_device()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype="auto",
    ).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print(f"Using device: {device}")
    print(f"Image: {image_path}")
    print("Ask questions about the image. Type 'exit' or 'quit' to stop.")

    messages = []
    is_first_question = True

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        user_message = make_user_message(
            question,
            image_path if is_first_question else None,
        )
        messages.append(user_message)

        answer = generate_answer(
            model,
            processor,
            messages,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"Qwen3-VL: {answer}")

        messages.append(make_assistant_message(answer))
        is_first_question = False

if __name__ == "__main__":
    main()
