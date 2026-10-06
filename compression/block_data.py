"""Tensor-local blocks and sampling conventions for learned-codec calibration.

NumPy only. Each tensor is padded independently. Stratum weights recover a
uniform full-block population from proportional samples with mandatory coverage.
"""
import numpy as np

BLOCK_SIZE = 256


def padded_blocks(values, block_size=BLOCK_SIZE):
    x=np.asarray(values)
    if x.dtype!=np.float32 or not np.isfinite(x).all() or type(block_size) is not int or block_size<1:
        raise ValueError('Expected finite float32 values and a positive block size')
    vector=x.reshape(-1)
    if not vector.size: raise ValueError('Empty tensors cannot be blocked')
    n=(vector.size+block_size-1)//block_size
    out=np.zeros((n,block_size),dtype=np.float32)
    out.reshape(-1)[:vector.size]=vector
    return out


def normalize_blocks(blocks):
    x=np.asarray(blocks)
    if x.dtype!=np.float32 or x.ndim!=2 or not np.isfinite(x).all():
        raise ValueError('Expected finite float32 block matrix')
    scales=np.sqrt(np.sum(x.astype(np.float64)**2,axis=1)).astype(np.float32)
    if not np.isfinite(scales).all(): raise ValueError('Block norm exceeds float32 range')
    result=np.zeros_like(x)
    np.divide(x,scales[:,None],out=result,where=scales[:,None]!=0)
    return result,scales


def stratified_counts(populations,budget):
    p=np.asarray(populations,dtype=np.int64)
    if p.ndim!=1 or not p.size or np.any(p<=0) or type(budget) is not int or budget<p.size:
        raise ValueError('Budget must cover every nonempty stratum')
    budget=min(budget,int(p.sum()))
    quotas=budget*p.astype(np.float64)/p.sum()
    counts=np.minimum(p,np.maximum(1,np.floor(quotas).astype(np.int64)))
    while counts.sum()>budget:
        scores=np.where(counts>1,counts-quotas,-np.inf)
        counts[int(np.argmax(scores))]-=1
    while counts.sum()<budget:
        scores=np.where(counts<p,quotas-counts,-np.inf)
        counts[int(np.argmax(scores))]+=1
    return counts


def snapshot_steps(batches,limit=10):
    if type(batches) is not int or batches<1 or type(limit) is not int or limit<1:
        raise ValueError('Positive batch count and snapshot limit required')
    n=min(batches,limit)
    return [(i*batches+n-1)//n for i in range(1,n+1)]
