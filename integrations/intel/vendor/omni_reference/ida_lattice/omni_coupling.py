# Portions derived from *The Hidden Canopy LLC* —
# [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
# Vendored from `src/ida_train/models/ida_lattice/omni_coupling.py`; body unmodified except as recorded in ../SOURCE.md.

"""Discrete, norm-bounded state-coordinate coupling and executable oracles.

U/V: [B,K,H,N,R], gamma: [B,K,H,R]. Factors may be shared by head
groups at their owner; expansion here is only the reference execution view.
No dense N x N delta matrix is constructed by either scan.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def select_program(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Correlated operation-program selection over the 8 joint A/B/C programs.

    logits: [..., 8] joint program logits, index = 4*a + 2*b + c (A major).
    Returns the [..., 3, 2] hard-forward straight-through mode one-hots and
    the [..., 3, 2] slot marginals of the 8-way softmax (a_i = sum_jk pi_ijk).
    With additively separable logits the softmax factorizes and the marginals
    coincide with the three independent per-slot softmaxes; the marginals are
    what carry the straight-through gradient into the joint logits.
    """
    if logits.shape[-1] != 8:
        raise ValueError("joint program logits must have width 8")
    compute_dtype = (torch.float32 if logits.dtype in (torch.float16, torch.bfloat16)
                     else logits.dtype)
    soft = logits.to(compute_dtype).softmax(-1).to(logits.dtype)
    index = soft.argmax(-1)
    bits = torch.stack((index >> 2, (index >> 1) & 1, index & 1), dim=-1)
    modes_hard = F.one_hot(bits, 2).to(soft.dtype)
    pi = soft.reshape(*soft.shape[:-1], 2, 2, 2)
    marginals = torch.stack((pi.sum(dim=(-2, -1)), pi.sum(dim=(-3, -1)),
                             pi.sum(dim=(-3, -2))), dim=-2)
    modes = modes_hard + (marginals - marginals.detach())
    return modes, marginals


def normalize_coupling(decay, left, right, gamma):
    dtype = torch.float64 if decay.dtype == torch.float64 else torch.float32
    decay, left, right, gamma = [x.to(dtype) for x in (decay, left, right, gamma)]
    column_bound = left.norm(dim=-2) * right.norm(dim=-2)
    coupling_bound = (gamma.abs() * column_bound).sum(-1)
    scale = (decay + coupling_bound[:, :, None]).clamp_min(1)
    return decay / scale, gamma[:, :, None] / scale[..., None], scale


def coupled_scan(u, b, c, decay, dt, weights, initial, mask, left, right, gamma, *, execution="packed"):
    if execution not in {"packed", "candidate", "native_cuda", "native_cuda_packed"}:
        raise ValueError("unknown coupled-state execution path")
    dtype = torch.float64 if initial.dtype == torch.float64 else torch.float32
    u,b,c,decay,dt,weights,initial,left,right,gamma = [x.to(dtype) for x in
        (u,b,c,decay,dt,weights,initial,left,right,gamma)]
    B,K,T,H,P = u.shape
    N,R = left.shape[-2:]
    if right.shape != left.shape or left.shape != (B,K,H,N,R) or gamma.shape != (B,K,H,R):
        raise ValueError("coupling factor geometry mismatch")
    if initial.shape != (B,H,P,N) or b.shape != (B,K,T,H,N) or c.shape != b.shape:
        raise ValueError("coupled state/input geometry mismatch")
    d, g, scale = normalize_coupling(decay,left,right,gamma)
    if execution in {"native_cuda", "native_cuda_packed"}:
        from .omni_native_scan import native_coupled_scan, native_packed_coupled_scan
        scan = native_coupled_scan if execution == "native_cuda" else native_packed_coupled_scan
        reads, state = scan(u,b,c,d,dt,weights,initial,mask,left,right,g)
        return reads,state,scale
    state=initial; reads=[]
    packed_v=right.permute(0,2,3,1,4).reshape(B,H,N,K*R)
    packed_u=left.permute(0,2,3,1,4).reshape(B,H,N,K*R)
    for t in range(T):
        bt,ct=b[:,:,t],c[:,:,t]
        if execution == "candidate":
            z=torch.matmul(state[:,None],right)
            delta=torch.matmul(z*g[:,:,t,:,None,:],left.transpose(-1,-2))
            candidate=(d[:,:,t,:,None,None]*state[:,None]+delta
                       +dt[:,:,t,:,None,None]*u[:,:,t,:,:,None]*bt[:,:,:,None,:])
            read=(candidate*ct[:,:,:,None,:]).sum(-1)
            merged=(candidate*weights[:,:,None,None,None]).sum(1)
        else:
            directions=torch.cat((packed_v,ct.permute(0,2,3,1)),dim=-1)
            query=state@directions
            z=query[...,:K*R].reshape(B,H,P,K,R).permute(0,3,1,2,4)
            hc=query[...,K*R:].permute(0,3,1,2)
            amplitude=z*g[:,:,t,:,None,:]
            uc=(left*ct[...,None]).sum(-2)
            bc=(bt*ct).sum(-1)
            read=(d[:,:,t,:,None]*hc+(amplitude*uc[:,:,:,None,:]).sum(-1)
                  +dt[:,:,t,:,None]*u[:,:,t]*bc[...,None])
            transport=(amplitude*weights[:,:,None,None,None]).permute(0,2,3,1,4).reshape(B,H,P,K*R)
            injection=(u[:,:,t]*dt[:,:,t,:,None]*weights[:,:,None,None]).permute(0,2,3,1)
            write_left=torch.cat((transport,injection),dim=-1)
            write_right=torch.cat((packed_u,bt.permute(0,2,3,1)),dim=-1)
            retention=(d[:,:,t]*weights[:,:,None]).sum(1)
            merged=retention[:,:,None,None]*state+write_left@write_right.transpose(-1,-2)
        valid=mask[:,t].bool()
        reads.append(torch.where(valid[:,None,None,None],read,0))
        state=torch.where(valid[:,None,None,None],merged,state)
    return torch.stack(reads,2),state,scale


class CoupledOperationControl(nn.Module):
    """Layer/expert-owned factors; no registration of the resident substrate."""
    def __init__(self, hidden, groups, state_dim, rank, program_selection="factorized"):
        super().__init__()
        if program_selection not in {"factorized", "correlated"}:
            raise ValueError("unknown Omni program selection")
        self.groups,self.rank=groups,rank
        self.program_selection=program_selection
        # Correlated selection emits 8 joint program logits instead of 6 slot
        # logits; the trailing groups*rank outputs are gamma in both modes.
        self.slot_outputs=8 if program_selection=="correlated" else 6
        outputs=self.slot_outputs+groups*rank
        self.current=nn.Linear(hidden,outputs)
        self.lrss=nn.Linear(hidden,outputs,bias=False)
        self.pss=nn.Linear(hidden,outputs,bias=False)
        self.error=nn.Linear(1,self.slot_outputs,bias=False)
        self.state_u=nn.Parameter(torch.empty(groups,state_dim,rank))
        self.state_v=nn.Parameter(torch.empty(groups,state_dim,rank))
        nn.init.normal_(self.state_u,std=state_dim**-.5)
        nn.init.normal_(self.state_v,std=state_dim**-.5)

    def forward(self,current,lrss,pss,error):
        from .omni_state import hard_binary
        sources=torch.stack((self.current(current),self.lrss(lrss),self.pss(pss)),1)
        total=sources.sum(1)
        bias=self.error(error[:,None].to(current.dtype))
        if self.program_selection=="correlated":
            modes,marginals=select_program(total[:,:8]+bias)
            # Contributions keep the [3, 6+groups*rank] contract: the 6 slot
            # entries hold the marginals of pi (source-agnostic, so repeated
            # per source row); the trailing entries stay per-source gamma.
            slot_marginals=marginals.reshape(*marginals.shape[:-2],6)
            contributions=torch.cat((slot_marginals.unsqueeze(1).expand(-1,3,-1),
                                     sources[...,8:]),dim=-1)
            return modes,total[:,8:],contributions
        modes=hard_binary((total[:,:6]+bias).reshape(-1,3,2))
        return modes,total[:,6:],sources
