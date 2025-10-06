"""
Unified Sampler Interface
Decouples generation logic from trainer and agent.
"""

from typing import List, Dict, Any, Union
import torch


class BaseSampler:
    """
    Base class for all samplers.
    Provides unified interface for generation.
    """
    
    def __init__(self, trainer):
        """
        Args:
            trainer: The GRPOTrainer instance
        """
        self.trainer = trainer
    
    def generate_and_score(
        self,
        inputs: List[Dict[str, Union[torch.Tensor, Any]]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        ⭐ UNIFIED SAMPLING INTERFACE ⭐
        
        Generate completions and compute scores for a batch of inputs.
        
        Args:
            inputs: List of input dicts, each containing:
                - prompt: conversation history (list of message dicts)
                - pixel_values: visual inputs (optional)
                - image_grid_thw: image grid info (optional)
                - ... other fields
        
        Returns:
            Dict containing:
                - completion_ids: generated token IDs
                - completion_mask: attention mask for completions
                - policy_per_token_logps: log probabilities from policy model
                - ... other fields required by GRPO
        """
        raise NotImplementedError("Subclasses must implement generate_and_score()")


class VLLMSampler(BaseSampler):
    """
    vLLM-based sampler (default for GRPOTrainer).
    Uses trainer's native vLLM generation.
    """
    
    def generate_and_score(
        self,
        inputs: List[Dict[str, Union[torch.Tensor, Any]]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Generate using vLLM (via trainer's _generate_and_score_completions).
        """
        # Call trainer's native generation method
        from trl import GRPOTrainer as _GRPOTrainer
        return _GRPOTrainer._generate_and_score_completions(self.trainer, inputs)
    
    def generate_from_prompts(
        self,
        prompts: List,
        images: List = None
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Generate from raw prompts (conversation list or strings).
        Uses trainer's _generate_single_turn.
        
        Args:
            prompts: List of conversation prompts (list of message dicts or strings)
            images: Optional list of images
        
        Returns:
            Dict containing completion_ids, completion_mask, etc.
        """
        # Call trainer's native single-turn generation
        from trl import GRPOTrainer as _GRPOTrainer
        return _GRPOTrainer._generate_single_turn(self.trainer, prompts, images)


class HuggingFaceSampler(BaseSampler):
    """
    HuggingFace transformers-based sampler.
    Uses model.generate() directly (slower but more flexible).
    """
    
    def generate_and_score(
        self,
        inputs: List[Dict[str, Union[torch.Tensor, Any]]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Generate using HuggingFace transformers.
        """
        # TODO: Implement HF generation logic
        raise NotImplementedError("HuggingFace sampler not implemented yet")


class BeamSearchSampler(BaseSampler):
    """
    Beam search sampler.
    """
    
    def __init__(self, trainer, num_beams: int = 4):
        super().__init__(trainer)
        self.num_beams = num_beams
    
    def generate_and_score(
        self,
        inputs: List[Dict[str, Union[torch.Tensor, Any]]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Generate using beam search.
        """
        # TODO: Implement beam search logic
        raise NotImplementedError("Beam search sampler not implemented yet")


def create_sampler(trainer, sampler_type: str = "vllm", **kwargs) -> BaseSampler:
    """
    Factory function to create sampler instances.
    
    Args:
        trainer: The GRPOTrainer instance
        sampler_type: Type of sampler ("vllm", "huggingface", "beam_search", etc.)
        **kwargs: Additional arguments for the sampler
    
    Returns:
        BaseSampler instance
    """
    samplers = {
        "vllm": VLLMSampler,
        "huggingface": HuggingFaceSampler,
        "beam_search": BeamSearchSampler,
    }
    
    if sampler_type not in samplers:
        raise ValueError(f"Unknown sampler type: {sampler_type}. Available: {list(samplers.keys())}")
    
    return samplers[sampler_type](trainer, **kwargs)

