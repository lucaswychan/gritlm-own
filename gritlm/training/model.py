from dataclasses import dataclass
import logging
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from transformers import AutoModel
from transformers.file_utils import ModelOutput
import numpy as np
import torch.nn.functional as F

from gritlm import GritLM

logger = logging.getLogger(__name__)


@dataclass
class GritLMTrainOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    loss_emb: Optional[Tensor] = None
    loss_gen: Optional[Tensor] = None


class DistributedContrastiveLoss:
    """Contrastive loss module that supports the standard InfoNCE objective as well as
    the *debiased* variant

    The implementation can optionally gather negatives across devices in a
    distributed setup so that every mini-batch benefits from a larger negative
    set without additional memory on a single GPU.
    """

    def __init__(
        self,
        temperature: float,
        negatives_cross_device: bool,
        *,
        debiased: bool = False,
        tau_plus: float = 0.1,
        cosine: bool = True,
    ):
        # Parameters shared by both loss variants
        self.temperature = temperature
        self.negatives_cross_device = negatives_cross_device

        # Parameters specific to debiased contrastive loss
        self.debiased = debiased
        self.tau_plus = tau_plus
        self.cosine = cosine

        # Components required only for the standard InfoNCE formulation
        if not self.debiased:
            self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")

        # Distributed setup
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError("Cannot use negatives_cross_device without distributed training")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        
        # Small caches to reduce per-step allocations
        self._gather_buf: Optional[torch.Tensor] = None
        self._arange_cache = {}

    def __call__(self, q_reps, p_reps):
        if self.negatives_cross_device:
            # Gather representations from all processes so that we can build a
            # larger negative set. Both positive and negative samples are
            # gathered; this could be optimised further but keeps the logic
            # simple.
            q_reps = self._dist_gather_tensor(q_reps)
            p_reps = self._dist_gather_tensor(p_reps)

        if self.debiased:
            return self._debiased_contrastive_loss(q_reps, p_reps)

        # ---------- Standard InfoNCE loss ----------
        scores = self.compute_similarity(q_reps, p_reps) / self.temperature
        # logger.info(f"scores in contrastive loss: {scores.shape}")
        scores = scores.view(q_reps.size(0), -1)

         # Cache arange to avoid re-allocation
        Bq = scores.size(0)
        base_idx = self._get_arange(Bq, scores.device)
        # Positive index stride within each block of passages
        target = base_idx * (p_reps.size(0) // q_reps.size(0))
    
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        if self.world_size == 1:
            return t
        t = t.contiguous()

        # Prefer faster gather-into-tensor if available, while preserving autograd behavior
        use_into_tensor = hasattr(dist, "all_gather_into_tensor")
        if use_into_tensor:
            local_bs = t.size(0)
            out_shape = (local_bs * self.world_size, *t.size()[1:])
            buf_mismatch = (
                self._gather_buf is None
                or tuple(self._gather_buf.shape) != out_shape
                or self._gather_buf.dtype != t.dtype
                or self._gather_buf.device != t.device
            )
            if buf_mismatch:
                self._gather_buf = torch.empty(out_shape, dtype=t.dtype, device=t.device)
            with torch.no_grad():
                dist.all_gather_into_tensor(self._gather_buf, t)
            # Build result by replacing local slice with the original tensor to keep gradients
            chunks = list(self._gather_buf.chunk(self.world_size, dim=0))
            chunks[self.rank] = t
            return torch.cat(chunks, dim=0)
        else:
            logger.info(f"Using list-based gather for dist_gather_tensor")
            # Fallback: list-based gather
            all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
            dist.all_gather(all_tensors, t)
            all_tensors[self.rank] = t
            return torch.cat(all_tensors, dim=0)

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    # ---------------------------------------------------------------------
    # Helper methods for the debiased contrastive loss
    # ---------------------------------------------------------------------
    def _debiased_contrastive_loss(self, q_reps: torch.Tensor, p_reps: torch.Tensor) -> torch.Tensor:
        """Assumes each query has *M* associated passages (`p_reps`) where the
        first passage in each consecutive block of size *M* is the positive
        example and the remaining *M-1* are negatives.
        """

        B = q_reps.size(0)
        # Determine group size M (passages per query)
        assert p_reps.size(0) % B == 0, "p_reps should contain an equal number of passages per query"
        M = p_reps.size(0) // B

        # Optionally normalise representations (cosine similarity)
        if self.cosine:
            q_norm = F.normalize(q_reps, dim=1)
            p_norm = F.normalize(p_reps, dim=1)
        else:
            q_norm = q_reps
            p_norm = p_reps

        # Similarity matrix between queries and ALL passages  [B, B*M]
        sim = torch.matmul(q_norm, p_norm.transpose(0, 1)) / self.temperature  # scaled by temperature
        
        # Compute row-wise log-sum-exp for numerical stability and reduced memory
        logsumexp_all = torch.logsumexp(sim, dim=1)  # [B]
        sum_exp_all = torch.exp(logsumexp_all)       # [B]

        # Positive indices are at i*M for query i
        base_idx = self._get_arange(B, sim.device)
        pos_logit = sim[base_idx, base_idx * M]      # [B]
        pos = torch.exp(pos_logit)                   # [B]

        # All negatives for each query
        neg = sum_exp_all - pos                      # [B]

        # De-biased estimator Ng
        if self.debiased:
            N = M * B - 1  # total negatives per query after concat across processes
            Ng = (-self.tau_plus * N * pos + neg) / (1.0 - self.tau_plus)
            Ng = torch.clamp(Ng, min=N * np.exp(-1.0 / self.temperature))
        else:
            Ng = neg

        loss = (-torch.log(pos / (pos + Ng))).mean()
        return loss

    @staticmethod
    def _get_masks(batch_size: int, device: torch.device):
        """Return boolean masks for negative and positive pairs.

        For a concatenated batch `[q_1..q_B, p_1..p_B]` of size `2B`, the
        positive for index *i* is `i+B` (if `i < B`) and `i-B` otherwise. All
        other indices are treated as negatives, except self-pairs which are
        excluded.
        """
        # Negative mask
        neg_mask = torch.ones((batch_size, 2 * batch_size), dtype=torch.bool, device=device)
        for i in range(batch_size):
            neg_mask[i, i] = False                 # q_i with q_i (self)
            neg_mask[i, i + batch_size] = False    # q_i with p_i (positive)
        neg_mask = torch.cat([neg_mask, neg_mask], dim=0)

        # Positive mask
        pos_mask = torch.zeros((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=device)
        idx = torch.arange(batch_size, device=device)
        pos_mask[idx, idx + batch_size] = True
        pos_mask[idx + batch_size, idx] = True

        return neg_mask, pos_mask

    def _get_arange(self, n: int, device: torch.device) -> torch.Tensor:
        key = (device.type, device.index)
        cached = self._arange_cache.get(key)
        if cached is None or cached.numel() < n:
            cached = torch.arange(n, device=device)
            self._arange_cache[key] = cached
        else:
            # Narrow without allocating when smaller n is requested
            cached = cached[:n]
        return cached

class NextTokenLoss:
    def __init__(self, vocab_size: int, loss_gen_type: str = "mixed", loss_gen_factor: float = 1.0):
        super().__init__()
        self.vocab_size = vocab_size
        # How to weight loss:
        # a) Each sample gets the same weight (e.g. used in BLOOMZ https://arxiv.org/abs/2211.01786)
        # -> This leads to shorter generations, as short outputs get the same weight as long ones
        # -> The loss curves for this are unstable if there's noisy or very hard short samples
        # b) Each token gets the same weight
        # -> This leads to longer generations, as long outputs get more weight than short ones
        # b.1) Each token gets the same weight globally across batches
        # -> Just sum the loss as is, optionally divide it by a constant. If using Adam, the scale
        # of the loss doesn't matter, so this is only to balance with another loss like in our case.
        # b.2) Each token gets the same weight per batch
        # -> Divide by the number of tokens in the batch
        # Problem: If distributed training, needs all gather of number of tokens on each process        
        # c) Mix of a) and b) which is what you do if you use the loss in transformers as is and 
        # then do grad acc/multi-gpu with bs>1 
        # (https://github.com/huggingface/transformers/issues/24725; https://github.com/pytorch/pytorch/issues/72047)
        self.loss_gen_factor = loss_gen_factor
        self.loss_gen_type = loss_gen_type
        if loss_gen_type == "token": # b.1)
            self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="sum")
        elif loss_gen_type == "mixed": # c)
            self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
        else:
            raise ValueError(f"Invalid loss_gen_type: {loss_gen_type}")
        
    def __call__(self, labels, logits):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        # Normalize by number of non-ignored tokens
        if self.loss_gen_type == "token":
            return (self.cross_entropy(shift_logits, shift_labels) / labels.size(0)) * self.loss_gen_factor
        elif self.loss_gen_type == "mixed":
            return self.cross_entropy(shift_logits, shift_labels) * self.loss_gen_factor


class GritLMTrainModel(GritLM):
    TRANSFORMER_CLS = AutoModel
    def __init__(
        self,
        temperature: float = 1.0,
        negatives_cross_device: bool = False,
        debiased: bool = False,
        tau_plus: float = 0.1,
        loss_gen_type: str = "mixed",
        loss_gen_factor: float = None,
        **kwargs,
    ):
        super().__init__(**kwargs, is_inference=False)
        self.emb_loss_fn = DistributedContrastiveLoss(temperature, negatives_cross_device, debiased=debiased, tau_plus=tau_plus)
        self.gen_add_kwargs = {"return_dict": True}
        if "mixtral" in kwargs["model_name_or_path"].lower():
            logger.info("Using token loss with routing loss for mixtral")
            self.gen_loss_fn = None
            self.gen_add_kwargs["loss_gen_factor"] = loss_gen_factor
            self.gen_add_kwargs["output_router_logits"] = True
        else:
            self.gen_loss_fn = NextTokenLoss(
                self.model.config.vocab_size, loss_gen_type, loss_gen_factor
            )
        self.config = self.model.config # Required for accelerate DeepSpeed integration

    def encode(self, features):
        if features is None: return None
        # Use in-place operations where safe to reduce memory allocations
        attention_mask = features.get('attention_mask')
        instruction_lens = features.get('instruction_lens')
        kwargs = {'input_ids': features.get('input_ids'), 'attention_mask': attention_mask}
        
        if self.attn[:2] == 'cb':
            kwargs['instruction_lens'] = instruction_lens
        elif self.attn[:2] == 'bb':
            kwargs['is_causal'] = False
        out = (getattr(self.model, self.embedding_attr) if self.embedding_attr else self.model)(**kwargs)[0]

        if self.projection is not None:
            out = self.projection(out)
        
        # Mask out the instruction tokens for pooling
        #@lucaswychan add checking of instruction_lens, since original approach will assume there is instruction in the passage
        if instruction_lens is not None and len(instruction_lens) > 0:
            # Clone only when necessary for instruction masking
            attention_mask = attention_mask.clone()
            # Vectorized masking for better performance
            batch_size = attention_mask.size(0)
            # Mask out the instruction tokens for pooling
            for i, l in enumerate(instruction_lens):
                if i < batch_size and l > 0:
                    attention_mask[i, :l] = 0
                    # Make sure not all zeros - If this happens it is a bug
                    assert attention_mask[i].sum() > 0, f"All 0: {attention_mask[i]}, l: {l}"


        reps = self.pooling(out, attention_mask)
        # Normalize can change the dtype (https://discuss.pytorch.org/t/tensor-in-float16-is-transformed-into-float32-after-torch-norm/110891)
        if self.normalized: 
            in_dtype = reps.dtype
            return torch.nn.functional.normalize(reps, dim=-1).contiguous().to(in_dtype)
        return reps.contiguous()

    def forward(
        self,
        query: Dict[str, torch.Tensor] = None,
        passage: Dict[str, torch.Tensor] = None,
        generative: Dict[str, torch.Tensor] = None,
        q_reps: Optional[torch.Tensor] = None,
        p_reps: Optional[torch.Tensor] = None,
        q_grad: bool = True,
        p_grad: bool = True,
    ):
        """
        Args:
            query: [b, n]
            passage: [b*s, m] where s is group size (usually 2)
            generative: [b, m]
        """
        # Do generative first, as emb contains an all-reduce (verified to be faster)
        if generative is not None:
            if self.gen_loss_fn is not None:
                # This pops the labels first, then the rest is passed into model                
                loss_gen = self.gen_loss_fn(
                    generative.pop('labels'), self.model(**generative, **self.gen_add_kwargs).logits
                )
            else:
                loss_gen = self.model(**generative, **self.gen_add_kwargs).loss
        else:
            loss_gen = None

        if (q_reps is None) and (query is not None):
            if q_grad:
                q_reps = self.encode(query)
            else:
                with torch.no_grad():
                    q_reps = self.encode(query)

        if (p_reps is None) and (passage is not None):
            if p_grad:
                p_reps = self.encode(passage)
            else:
                with torch.no_grad():
                    p_reps = self.encode(passage)
            
        loss_emb = self.emb_loss_fn(
            q_reps, p_reps
        ) if (q_reps is not None and p_reps is not None) else None        

        # Optimize loss combination
        if loss_emb is not None and loss_gen is not None:
            loss = loss_emb + loss_gen
        elif loss_emb is not None:
            loss = loss_emb
        elif loss_gen is not None:
            loss = loss_gen
        else:
            loss = None

        # Also return q_reps in case of GradCache
        return GritLMTrainOutput(
            q_reps=q_reps,
            p_reps=p_reps,
            loss=loss,
            loss_emb=loss_emb,
            loss_gen=loss_gen,
        )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)