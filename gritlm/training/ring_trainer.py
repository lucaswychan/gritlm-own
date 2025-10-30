"""
Ring-based Trainer for efficient distributed contrastive learning.

This trainer uses ring communication patterns to compute contrastive loss
across multiple GPUs without gathering all embeddings, avoiding the memory
overhead and complexity of GradCache.
"""

import contextlib
import functools
import logging
from typing import Dict, Optional

import torch
import torch.distributed as dist
from transformers import Trainer
from transformers.trainer_pt_utils import find_batch_size

from .ring_loss import cal_inf_loss, cal_ring_loss

logger = logging.getLogger(__name__)


class RingTrainer(Trainer):
    """
    Trainer that uses ring-based communication for efficient contrastive loss.
    
    This avoids the need for GradCache and works seamlessly with FSDP.
    Only supports embedding mode.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute embedding loss using ring-based contrastive loss.
        
        Args:
            model: The model to train
            inputs: Dict containing 'query' and 'passage' keys with tokenized inputs
            return_outputs: Whether to return model outputs along with loss
            
        Returns:
            loss or (loss, outputs) tuple
        """
        return self._compute_embedding_loss(model, inputs, return_outputs)
    
    def _compute_embedding_loss(self, model, inputs, return_outputs=False):
        """
        Compute embedding loss using ring-based contrastive learning.
        
        This encodes queries and passages, then uses ring communication
        to compute the contrastive loss across all GPUs efficiently.
        """
        # Encode queries and passages
        query_inputs = inputs.get('query')
        passage_inputs = inputs.get('passage')
        
        if query_inputs is None or passage_inputs is None:
            raise ValueError("Both 'query' and 'passage' must be in inputs for embedding mode")
        
        # Forward pass through the model
        outputs = model(query=query_inputs, passage=passage_inputs)
        q_reps = outputs.q_reps  # [batch_size, hidden_dim]
        p_reps = outputs.p_reps  # [batch_size * group_size, hidden_dim]
        
        # Get batch size
        batch_size = q_reps.size(0)
        group_size = p_reps.size(0) // batch_size
        
        # Prepare labels - for each query, the positive is at index i * group_size
        # where i is the query index in the current batch
        if dist.is_initialized():
            # In distributed setting, need to account for rank offset
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_batch_size = batch_size
            # Labels point to the positive passage in the global passage tensor
            labels = torch.arange(batch_size, device=q_reps.device) * group_size + rank * local_batch_size * group_size
        else:
            # Single GPU - labels are just indices of positives
            labels = torch.arange(batch_size, device=q_reps.device) * group_size
        
        # Scale by temperature
        scale = torch.tensor(1.0 / self.temperature, device=q_reps.device)
        
        # Use ring-based or inf-based loss
        if self.use_inf_loss:
            loss_fn = cal_inf_loss
        else:
            loss_fn = cal_ring_loss
        
        # Compute bidirectional contrastive loss
        # Query-to-passage loss
        loss_q2p = loss_fn(q_reps, p_reps, labels=labels, scale=scale, head_dim=self.head_dim)
        
        # Passage-to-query loss (symmetry)
        # Note: We only use positives for p2q, so extract them
        p_pos = p_reps[labels - (rank * local_batch_size * group_size if dist.is_initialized() else 0)]
        loss_p2q = loss_fn(p_pos, q_reps, labels=torch.arange(batch_size, device=q_reps.device), 
                          scale=scale, head_dim=self.head_dim)
        
        # Combined loss
        loss = (loss_q2p + loss_p2q) * 0.5
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step on a batch of inputs.
        
        This is simpler than GradCache since ring loss handles distribution automatically.
        
        Args:
            model: The model to train
            inputs: The inputs for the model
            num_items_in_batch: Number of items in the batch (used for gradient accumulation)
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        
        # Scale loss for gradient accumulation
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        self.accelerator.backward(loss)
        
        return loss.detach()
    
    def _prepare_input(self, data):
        """
        Prepares one input for the model, handling nested structures.
        Override to ensure proper device placement.
        """
        if isinstance(data, dict):
            return {k: self._prepare_input(v) for k, v in data.items()}
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            # Don't put back to half precision if it was casted to fp32
            if data.dtype != torch.int64 and data.dtype != torch.int32:
                kwargs["dtype"] = self.args_to_dtype()
            return data.to(**kwargs)
        return data
    
    def args_to_dtype(self):
        """Get the target dtype for mixed precision training."""
        if self.args.bf16:
            return torch.bfloat16
        if self.args.fp16:
            return torch.float16
        return torch.float32


def create_ring_trainer(
    model,
    args,
    train_dataset,
    data_collator,
    tokenizer,
    temperature=0.02,
    use_inf_loss=True,
    head_dim=256,
    **kwargs
):
    """
    Factory function to create a RingTrainer for embedding training.
    
    Args:
        model: The model to train
        args: Training arguments
        train_dataset: Training dataset
        data_collator: Data collator
        tokenizer: Tokenizer
        temperature: Temperature for contrastive loss (default: 0.02)
        use_inf_loss: Whether to use InfProb (Flash-based, faster) vs RingProb (default: True)
        head_dim: Head dimension for ring loss (must be 16, 32, 64, 128, or 256)
        **kwargs: Additional arguments to pass to trainer
        
    Returns:
        RingTrainer instance
    """
    trainer = RingTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        **kwargs
    )
    # Set ring-specific attributes
    trainer.temperature = temperature
    trainer.use_inf_loss = use_inf_loss
    trainer.head_dim = head_dim
    return trainer

