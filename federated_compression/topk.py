"""Global Top-k model-delta adapter with persistent per-client error feedback.

State belongs to Flower Context, not a transient ClientApp instance. A lost state,
duplicate round, or incompatible state fails closed. Interrupted runs restart
from round one in a new run; model-only checkpoints cannot resume residuals.
"""
import hashlib
import json
from time import perf_counter
import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, MetricRecord
from compression import topk

WIRE = 'topk-f32-u32-ef-v1'


def layout(reference):
    entries=[]
    for name in sorted(reference):
        value=reference[name]
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Top-k requires finite float32 model tensors')
        entries.append([name, list(value.shape), str(value.dtype)])
    digest=hashlib.sha256(json.dumps(entries,separators=(',',':')).encode()).hexdigest()
    return digest, sum(v.numel() for v in reference.values())


def metadata(reference, k, server_round):
    digest,n=layout(reference)
    if type(k) is not int or not 1 <= k <= n:
        raise ValueError('topk-k exceeds trainable coordinates or is invalid')
    return dict(codec=WIRE, layout=digest, size=n, k=k, **{'server-round':server_round})


def encode(trained, reference, context, *, k, server_round, client_id):
    start=perf_counter()
    if set(trained)!=set(reference):
        raise ValueError('Top-k tensor names differ')
    meta=metadata(reference,k,server_round)
    state_meta=dict(meta, run_id=context.run_id, client_id=client_id)
    if server_round == 1:
        if 'topk-meta' in context.state or 'topk-residual' in context.state:
            raise ValueError('Unexpected existing Top-k state in first round')
        residual=np.zeros(meta['size'],dtype=np.float32)
    else:
        expected=dict(state_meta, **{'server-round':server_round-1})
        if 'topk-meta' not in context.state or dict(context.state['topk-meta']) != expected:
            raise ValueError('Missing, stale, or incompatible Top-k residual state')
        residual=context.state['topk-residual'].to_numpy_ndarrays()[0]
        if residual.shape!=(meta['size'],) or residual.dtype!=np.float32 or not np.isfinite(residual).all():
            raise ValueError('Invalid Top-k residual')
    parts=[]
    for name in sorted(reference):
        value=trained[name].detach().cpu(); base=reference[name].detach().cpu()
        if value.shape!=base.shape or value.dtype!=base.dtype:
            raise ValueError('Top-k tensor shape/dtype mismatch')
        parts.append((value-base).numpy().reshape(-1))
    delta=np.concatenate(parts)
    corrected=delta+residual
    packet=topk.encode(corrected,k)
    reconstructed=topk.decode(packet,expected_size=meta['size'],expected_k=k)
    new_residual=corrected-reconstructed
    context.state['topk-residual']=ArrayRecord([new_residual])
    context.state['topk-meta']=ConfigRecord(state_meta)
    encode_seconds=perf_counter()-start
    diagnostic_start=perf_counter()
    norm=lambda x: float(np.dot(x.astype(np.float64),x.astype(np.float64)))
    stats=dict(topk_encode_seconds=encode_seconds,
               topk_corrected_squared_norm=norm(corrected),
               topk_residual_squared_norm=norm(new_residual),
               topk_delta_squared_norm=norm(delta),
               topk_transmitted_vs_delta_squared_error=norm(reconstructed-delta))
    stats['topk_diagnostic_seconds']=perf_counter()-diagnostic_start
    return ConfigRecord({'packet':packet}),ConfigRecord(meta),stats


def decode(content, reference, *, k, server_round):
    if set(content)!={'topk','update','metrics','client'}:
        raise ValueError('Unexpected Top-k records')
    expected=metadata(reference,k,server_round)
    meta=content['update']
    if not isinstance(meta,ConfigRecord) or dict(meta)!=expected or any(type(meta[a]) is not type(b) for a,b in expected.items()):
        raise ValueError('Top-k metadata mismatch')
    packets=content['topk']; metrics=content['metrics']
    if not isinstance(packets,ConfigRecord) or set(packets)!={'packet'} or not isinstance(metrics,MetricRecord):
        raise ValueError('Invalid Top-k records')
    if type(metrics.get('num-examples')) is not int or metrics['num-examples']<=0:
        raise ValueError('Invalid sample count')
    if not isinstance(metrics.get('train_loss'),(int,float)) or not np.isfinite(metrics['train_loss']):
        raise ValueError('Invalid training loss')
    vector=topk.decode(packets['packet'],expected_size=expected['size'],expected_k=k)
    result={}; offset=0
    for name in sorted(reference):
        base=reference[name]; n=base.numel()
        result[name]=torch.from_numpy(vector[offset:offset+n].reshape(tuple(base.shape)).copy())
        offset+=n
    return result
