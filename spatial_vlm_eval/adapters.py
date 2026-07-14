from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL import Image


class VLMAdapter(Protocol):
    def ask(self, image_paths: list[str | Path], question: str, max_new_tokens: int = 40) -> str:
        ...


class MockAdapter:
    """Deterministic adapter for pipeline tests.

    It reads the encoded ground truth when provided by the evaluator. This keeps
    CI and local smoke tests independent from GPU/model downloads.
    """

    def __init__(self) -> None:
        self.next_answer: str | None = None

    def set_next_answer(self, answer: str) -> None:
        self.next_answer = answer

    def ask(self, image_paths: list[str | Path], question: str, max_new_tokens: int = 40) -> str:
        if self.next_answer is None:
            return "unknown"
        return self.next_answer


class SmolVLMAdapter:
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.float_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=self.float_dtype).to(self.device)
        self.model.eval()

    def ask(self, image_paths: list[str | Path], question: str, max_new_tokens: int = 40) -> str:
        import torch

        if not image_paths:
            raise ValueError("At least one image is required.")
        images = [Image.open(path).convert("RGB") for path in image_paths]
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": question}],
            }
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=images, return_tensors="pt")
        moved = {}
        for key, value in inputs.items():
            if value.is_floating_point():
                moved[key] = value.to(self.device, dtype=self.float_dtype)
            else:
                moved[key] = value.to(self.device)
        input_length = moved["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = self.model.generate(**moved, max_new_tokens=max_new_tokens, do_sample=False)
        generated = output_ids[:, input_length:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


class Qwen25VLAdapter:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.float_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=self.float_dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

    def ask(self, image_paths: list[str | Path], question: str, max_new_tokens: int = 40) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        if not image_paths:
            raise ValueError("At least one image is required.")

        content = []
        for image_path in image_paths:
            content.append({"type": "image", "image": Image.open(image_path).convert("RGB")})
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        input_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = output_ids[:, input_length:]
        return self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def build_adapter(name: str, model_id: str | None = None) -> VLMAdapter:
    if name == "mock":
        return MockAdapter()
    if name == "smolvlm":
        return SmolVLMAdapter(model_id=model_id or "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    if name == "qwen2_5_vl":
        return Qwen25VLAdapter(model_id=model_id or "Qwen/Qwen2.5-VL-3B-Instruct")
    raise ValueError(f"Unsupported adapter: {name}")
