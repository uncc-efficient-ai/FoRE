import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import (  # FORE-TARGET: + AutoConfig and the multimodal loader
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
)


@dataclass
class DFlashTargetOutput:
    hidden_states: torch.Tensor  # [batch, seq_len, hidden_size]
    input_ids: torch.Tensor  # [batch, seq_len]
    attention_mask: torch.Tensor  # [batch, seq_len]
    loss_mask: torch.Tensor  # [batch, seq_len]


class DFlashTargetModel(ABC):
    """
    Abstract base class for DFlash target model backend.
    """

    def __init__(self):
        self.capture_layer_ids = None

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "DFlashTargetModel":
        """Initialize the target model backend."""

    @abstractmethod
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DFlashTargetOutput:
        """Generate context hidden states for DFlash training."""

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        """Set which layers' hidden states to capture."""
        self.capture_layer_ids = layer_ids


class HFDFlashTargetModel(DFlashTargetModel):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "HFDFlashTargetModel":

        # FORE-TARGET: multimodal targets (e.g. Qwen3.5-9B = Qwen3_5ForConditionalGeneration)
        # load via AutoModelForImageTextToText, not AutoModelForCausalLM.
        cfg = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code
        )
        archs = cfg.architectures or []
        is_multimodal = any(
            "ConditionalGeneration" in a or "ImageTextToText" in a for a in archs
        )
        loader = AutoModelForImageTextToText if is_multimodal else AutoModelForCausalLM
        target_model = loader.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            output_hidden_states=True,
            trust_remote_code=trust_remote_code,
            **kwargs,
        ).eval()

        if device:
            target_model = target_model.to(device)

        # FORE-TARGET: the drafter only consumes text hidden states and the forward never
        # receives pixel_values, so the vision tower is dead weight. Drop it to free its GPU
        # memory.
        if is_multimodal:
            vision_owner = getattr(target_model, "model", target_model)
            if getattr(vision_owner, "visual", None) is not None:
                vision_owner.visual = None
                gc.collect()  # the tower sits in a reference cycle; empty_cache alone won't reclaim it
                if device and "cuda" in str(device):
                    torch.cuda.empty_cache()

        return cls(target_model)

    @torch.no_grad()
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DFlashTargetOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # hidden_states[0] = embedding output; hidden_states[i+1] = layer i output
        offset = 1
        selected = []
        if self.capture_layer_ids is not None:
            for idx in self.capture_layer_ids:
                selected.append(outputs.hidden_states[idx + offset])
            hidden_states = torch.cat(selected, dim=-1)
        else:
            hidden_states = outputs.hidden_states[-1]

        return DFlashTargetOutput(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )


def get_dflash_target_model(
    pretrained_model_name_or_path: str,
    backend: str = "hf",
    torch_dtype: torch.dtype = None,
    device: str = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> DFlashTargetModel:
    if backend == "hf":
        return HFDFlashTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")
