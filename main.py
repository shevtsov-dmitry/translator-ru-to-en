#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "Helsinki-NLP/opus-mt-ru-en"
MAX_INPUT_TOKENS = 450


def split_into_chunks(text, tokenizer, max_tokens):
    """Split text into chunks that fit within the model's input limit."""

    token_ids = tokenizer.encode(text, add_special_tokens=False)

    chunks = []

    for i in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[i : i + max_tokens]

        chunks.append(
            tokenizer.decode(
                chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
        )

    return chunks


def translate_block(text, tokenizer, model, device):
    """Translate a text block, splitting it if necessary."""

    chunks = split_into_chunks(text, tokenizer, MAX_INPUT_TOKENS)

    translated = []

    for chunk in chunks:
        if not chunk.strip():
            translated.append(chunk)
            continue

        inputs = tokenizer(
            chunk, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs, num_beams=4, max_new_tokens=512, early_stopping=True
            )

        result = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        translated.append(result)

    return " ".join(translated)


def translate_text(text, tokenizer, model, device):
    """Translate text while preserving paragraph separators."""

    # Capture paragraph separators instead of discarding them.
    parts = re.split(r"(\r\n[ \t]*\r\n+|\n[ \t]*\n+|\r[ \t]*\r+)", text)

    translated_parts = []

    total = sum(1 for part in parts if part.strip())

    completed = 0

    for part in parts:
        if not part.strip():
            translated_parts.append(part)
            continue

        # Preserve leading and trailing whitespace.
        leading = part[: len(part) - len(part.lstrip())]
        trailing = part[len(part.rstrip()) :]
        content = part.strip()

        result = translate_block(content, tokenizer, model, device)

        translated_parts.append(leading + result + trailing)

        completed += 1

        print(f"\rTranslating: {completed}/{total} paragraphs", end="", flush=True)

    if total:
        print()

    return "".join(translated_parts)


def main():
    parser = argparse.ArgumentParser(
        description="Translate Russian text files into English using Hugging Face."
    )

    parser.add_argument("file", type=Path, help="Input text file (TXT, MD, INI, etc.)")

    args = parser.parse_args()

    input_path = args.file

    if not input_path.is_file():
        sys.exit(f"Error: File not found: {input_path}")

    # Require NVIDIA CUDA.
    if not torch.cuda.is_available():
        sys.exit(
            "Error: CUDA is not available. "
            "Install a CUDA-enabled PyTorch build and check your NVIDIA driver."
        )

    device = torch.device("cuda")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    # Construct output filename.
    output_path = input_path.with_name(f"{input_path.stem}(EN){input_path.suffix}")

    # Read input, preserving original line endings.
    with open(input_path, "r", encoding="utf-8-sig", newline="") as file:
        original_text = file.read()

    if not original_text.strip():
        sys.exit("Error: Input file is empty.")

    print(f"\nLoading model: {MODEL_NAME}")

    # Download automatically on first execution.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)

    model.to(device)
    model.eval()

    print("Model loaded successfully.")
    print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}\n")

    translated_text = translate_text(original_text, tokenizer, model, device)

    # Save translated file.
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        file.write(translated_text)

    print("\nTranslation completed!")
    print(f"Saved: {output_path}")

    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
