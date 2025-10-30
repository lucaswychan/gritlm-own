"""
Ring-based contrastive loss for efficient distributed training.

Based on Ring Attention communication patterns to compute contrastive loss
across multiple GPUs without gathering all embeddings.
"""

import os
import math
import random

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import torch.nn.functional as F
import numpy as np


class RingComm:
    """Ring communication for passing tensors between GPUs in a ring topology."""

    def __init__(self, process_group: dist.ProcessGroup = None):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group) if process_group else dist.get_rank()
        self.world_size = dist.get_world_size(self._process_group) if process_group else dist.get_world_size()
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size
        
        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(self, to_send, recv_tensor=None):
        """Send tensor to next rank and receive from previous rank."""
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        """Commit all pending send/recv operations."""
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        """Wait for all pending operations to complete."""
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []


class GradientGather(torch.autograd.Function):
    """Gather gradients across all ranks for scaling factor."""
    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, dx):
        if dist.is_initialized():
            dist.all_reduce(dx)
        return dx


class RingProb(torch.autograd.Function):
    """
    Ring-based probability computation for contrastive loss.
    
    Computes log-sum-exp over all keys across GPUs using ring communication.
    """

    @staticmethod
    def forward(ctx, q, k, group):
        k = k.contiguous()
        comm = RingComm(group)

        colle = [q, k]

        lse = None
        next_k = None
        for step in range(comm.world_size):
            if step + 1 != comm.world_size:
                next_k = comm.send_recv(k)
                comm.commit()

            # Vanilla LSE computation
            qk = torch.einsum("mhd,nhd->mn", q, k)
            block_lse = torch.log(torch.exp(qk).sum(dim=-1))

            if step == 0:
                lse = block_lse
            else:
                lse = lse - F.logsigmoid(lse - block_lse)

            if step + 1 != comm.world_size:
                comm.wait()
                k = next_k

        colle.append(lse)
        ctx.save_for_backward(*colle)
        ctx.group = group
        return lse

    @staticmethod
    def backward(ctx, dlse):
        q, k, lse = ctx.saved_tensors
        k_comm = RingComm(ctx.group)
        d_k_comm = RingComm(ctx.group)
        dq, dk = None, None
        next_dk = None

        block_dq_buffer = torch.empty(q.shape, dtype=torch.float32, device=q.device)
        block_dk_buffer = torch.empty(k.shape, dtype=torch.float32, device=k.device)

        next_dk, next_k = None, None

        for step in range(k_comm.world_size):
            if step + 1 != k_comm.world_size:
                next_k = k_comm.send_recv(k)
                k_comm.commit()

            # Vanilla gradient calculation
            qk = torch.einsum("mhd,nhd->mn", q, k)
            qk_grad = torch.exp(qk - lse[:, None]).float()
            qk_grad = qk_grad * dlse[:, None]
            block_dq_buffer = torch.einsum("mn,nhd->mhd", qk_grad, k.float())
            block_dk_buffer = torch.einsum("nm,mhd->nhd", qk_grad.T, q.float())

            if step == 0:
                dq = block_dq_buffer
                dk = block_dk_buffer
            else:
                dq += block_dq_buffer
                d_k_comm.wait()
                dk = block_dk_buffer + next_dk

            if step + 1 != k_comm.world_size:
                k_comm.wait()
                k = next_k

            next_dk = d_k_comm.send_recv(dk)
            d_k_comm.commit()

        d_k_comm.wait()

        return dq, next_dk, None


# Placeholder for InfProb - requires flash attention implementation
# For now, we'll use RingProb for both
class InfProb(torch.autograd.Function):
    """
    Flash-optimized ring probability computation.
    
    Note: This requires the flash attention kernels from the original implementation.
    Falls back to RingProb if flash kernels are not available.
    """

    @staticmethod
    def forward(ctx, q, k, group):
        # Try to use flash implementation if available
        try:
            from .flash import _flash_prob_forward
            
            k = k.contiguous()
            comm = RingComm(group)

            colle = [q, k]

            lse = None
            next_k = None
            for step in range(comm.world_size):
                if step + 1 != comm.world_size:
                    next_k = comm.send_recv(k)
                    comm.commit()

                # Flash LSE
                block_lse = _flash_prob_forward(q, k)

                if step == 0:
                    lse = block_lse
                else:
                    lse = lse - F.logsigmoid(lse - block_lse)

                if step + 1 != comm.world_size:
                    comm.wait()
                    k = next_k

            colle.append(lse)
            ctx.save_for_backward(*colle)
            ctx.group = group
            return lse
        except ImportError:
            # Fallback to RingProb
            return RingProb.forward(ctx, q, k, group)

    @staticmethod
    def backward(ctx, dlse):
        try:
            from .flash import _flash_prob_backward
            
            q, k, lse = ctx.saved_tensors
            k_comm = RingComm(ctx.group)
            d_k_comm = RingComm(ctx.group)
            dq, dk = None, None
            next_dk = None

            block_dq_buffer = torch.empty(q.shape, dtype=torch.float32, device=q.device)
            block_dk_buffer = torch.empty(k.shape, dtype=torch.float32, device=k.device)

            next_dk, next_k = None, None

            for step in range(k_comm.world_size):
                if step + 1 != k_comm.world_size:
                    next_k = k_comm.send_recv(k)
                    k_comm.commit()

                # Flash gradient calculation
                block_dq_buffer, block_dk_buffer = _flash_prob_backward(q, k, lse, dlse)

                if step == 0:
                    dq = block_dq_buffer
                    dk = block_dk_buffer
                else:
                    dq += block_dq_buffer
                    d_k_comm.wait()
                    dk = block_dk_buffer + next_dk

                if step + 1 != k_comm.world_size:
                    k_comm.wait()
                    k = next_k

                next_dk = d_k_comm.send_recv(dk)
                d_k_comm.commit()

            d_k_comm.wait()

            return dq, next_dk, None
        except ImportError:
            # Fallback to RingProb
            return RingProb.backward(ctx, dlse)


def _cal_ring_loss(q, k, labels, head_dim=256):
    """Internal function to compute ring loss."""
    bq = q.shape[0]
    bk = k.shape[0]
    q = q.view(bq, -1, head_dim).float()
    k = k.view(bk, -1, head_dim).float()

    lse = RingProb.apply(q, k, None)
    numerator = torch.einsum("mhd,mhd->m", q, k[labels, ...])
    loss = -numerator + lse

    return loss


def _cal_inf_loss(q, k, labels, head_dim=256):
    """Internal function to compute inf loss (flash-optimized)."""
    bq = q.shape[0]
    bk = k.shape[0]
    q = q.view(bq, -1, head_dim).float()
    k = k.view(bk, -1, head_dim).float()

    lse = InfProb.apply(q, k, None)
    numerator = torch.einsum("mhd,mhd->m", q, k[labels, ...])
    loss = -numerator + lse

    return loss


def _cal_flash_loss(q, k, labels, head_dim=256):
    """Fallback single-GPU flash loss."""
    bq = q.shape[0]
    bk = k.shape[0]
    q = q.view(bq, -1, head_dim).float()
    k = k.view(bk, -1, head_dim).float()

    # Simple contrastive loss for single GPU
    qk = torch.einsum("mhd,nhd->mn", q, k)
    loss = F.cross_entropy(qk, labels, reduction='none')
    
    return loss


def cal_ring_loss(q, k, labels=None, scale=None, head_dim=256):
    """
    The triton implementation of the ring-cl.

    Args:
        q (torch.Tensor): The column tensor in contrastive loss. The shape is [B, D].
        k (torch.Tensor): The row tensor in contrastive loss. The shape is [B, D].
        labels (torch.Tensor, optional): Indices of positive pairs. Shape is [B]. 
            When None, labels are range(B). Defaults to None.
        scale (torch.Tensor, optional): Scale tensor for query. Defaults to None.
        head_dim (int, optional): Head dimension (must be 16, 32, 64, 128 or 256). 
            Defaults to 256.

    Returns:
        torch.Tensor: Computed loss
    """
    if labels is None:
        labels = torch.arange(q.shape[0], device=q.device)
    if scale is None:
        scale = 1.0
    else:
        scale = GradientGather.apply(scale)
    
    if dist.is_initialized():
        return _cal_ring_loss(scale * q, k, labels, head_dim).mean()
    else:
        return _cal_flash_loss(scale * q, k, labels, head_dim).mean()


def cal_inf_loss(q, k, labels=None, scale=None, head_dim=256):
    """
    The triton implementation of the inf-cl (flash-optimized).

    Args:
        q (torch.Tensor): The column tensor in contrastive loss. The shape is [B, D].
        k (torch.Tensor): The row tensor in contrastive loss. The shape is [B, D].
        labels (torch.Tensor, optional): Indices of positive pairs. Shape is [B].
            When None, labels are range(B). Defaults to None.
        scale (torch.Tensor, optional): Scale tensor for query. Defaults to None.
        head_dim (int, optional): Head dimension (must be 16, 32, 64, 128 or 256).
            Defaults to 256.

    Returns:
        torch.Tensor: Computed loss
    """
    if labels is None:
        labels = torch.arange(q.shape[0], device=q.device)
    if scale is None:
        scale = 1.0
    else:
        scale = GradientGather.apply(scale)
    
    if dist.is_initialized():
        return _cal_inf_loss(scale * q, k, labels, head_dim).mean()
    else:
        return _cal_flash_loss(scale * q, k, labels, head_dim).mean()

