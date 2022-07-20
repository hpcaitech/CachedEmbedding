from functools import partial
from contextlib import contextmanager

import pytest
import torch 
import numpy as np

from recsys.modules.embeddings import LoadBalanceManager
from common import  NUM_EMBEDDINGS_PER_FEATURE, BATCH_SIZE


def check_loadbalancemanager(world_size, device):
    # print(f'setting: world_size:{world_size}, device:{device}')
    lbmgr = LoadBalanceManager(embeddings_per_feat=NUM_EMBEDDINGS_PER_FEATURE,
                               num_groups=world_size,
                               do_fair=True,
                               device=device)
    
    rand_input = []
    
    for i in range(len(NUM_EMBEDDINGS_PER_FEATURE)):
        rand_input.append(torch.randint(0,NUM_EMBEDDINGS_PER_FEATURE[i],
                                               size=(BATCH_SIZE,)).unsqueeze(1))
        
    rand_input_tensor = torch.cat(rand_input,dim=1).to(device)
    
    assert rand_input_tensor.shape == (BATCH_SIZE, len(NUM_EMBEDDINGS_PER_FEATURE))
    
    # find the sharding points / cut positions and offsets for each shard rank
    dim_indices = np.array(range(len(NUM_EMBEDDINGS_PER_FEATURE)))
    num_embeddings_per_rank = sum(NUM_EMBEDDINGS_PER_FEATURE) // world_size
    cuts = {}
    offsets = []
    _curr_offs = [0]
    _num_cuts = 0
    _agg = num_embeddings_per_rank
    
    for ind in dim_indices:
        while NUM_EMBEDDINGS_PER_FEATURE[ind] > _agg:
            if _num_cuts >= world_size - 1: # never cut when enough groups
                break
            if ind in cuts.keys():
                cuts[ind].append(_agg)
            else:
                cuts[ind] = [_agg]
            _num_cuts += 1
            
            offsets.append(torch.tensor(_curr_offs,device=device))
            _curr_offs = [0]
            _agg += num_embeddings_per_rank
        
        if _agg >= NUM_EMBEDDINGS_PER_FEATURE[ind] and len(_curr_offs) == 1:
            _curr_offs.append(NUM_EMBEDDINGS_PER_FEATURE[ind]-(_agg-num_embeddings_per_rank))
        else:
            _curr_offs.append(NUM_EMBEDDINGS_PER_FEATURE[ind])
        
        _agg -= NUM_EMBEDDINGS_PER_FEATURE[ind]
    
    offsets.append(torch.tensor(_curr_offs[:-1],device=device))
    for i in range(len(offsets)):
        offsets[i] = torch.cumsum(offsets[i], dim=0)
    
    for (a,b) in zip(offsets, lbmgr.offsets):
        assert torch.allclose(a,b)
    
    # Perform tensor sharding
    for rank in range(world_size):
        test_shard_output = lbmgr.shard_tensor(rand_input_tensor.clone(),rank)
        offset = offsets[rank]
        _cinput = rand_input_tensor.clone()
        if world_size == 1: # no cut
            shard_output = _cinput + offsets[0]
            assert torch.allclose(test_shard_output,shard_output)
            continue
        feats = list(cuts.keys())
        if rank == 0:
            feat_id = feats[0]
            cut_pos = cuts[feat_id][0]
            _cinput[:,feat_id] = torch.min((cut_pos-offset[-1]-1)*torch.ones(_cinput.size(0),device=device),\
                                            _cinput[:,feat_id])                
            shard_output = _cinput[:,:feat_id+1] + offset
            assert torch.allclose(test_shard_output,shard_output)
            continue
        else:
            rank -= 1
            for (k,v) in cuts.items():
                if rank - len(v) < 0:
                    cut_pos = v[rank:][:2] # this and next shard position
                    if len(cut_pos) < 2 and k != feats[-1]:
                        next_feat_id = feats[feats.index(k) + 1]
                    else:
                        next_feat_id = None
                    feat_id = k
                    break
                rank -= len(v)
            if len(cut_pos) == 1:
                cut_pos = cut_pos[0]
                if feat_id == feats[-1]: # last rank
                    _cinput[:,feat_id] = torch.max(torch.zeros(_cinput.size(0),device=device), _cinput[:,feat_id]-cut_pos)
                    shard_output = _cinput[:,feat_id:] + offset
                    assert torch.allclose(test_shard_output,shard_output)
                    continue
                else:
                    assert next_feat_id is not None
                    _cinput[:,feat_id] = torch.max(torch.zeros(_cinput.size(0),device=device), 
                                                _cinput[:,feat_id]-cut_pos)
                    _cinput[:,next_feat_id] = torch.min((cut_pos-offset[-1]-1)*torch.ones(_cinput.size(0),device=device), 
                                                        _cinput[:,next_feat_id])
                    shard_output = _cinput[:,feat_id:next_feat_id+1] + offset
                    assert torch.allclose(test_shard_output,shard_output)
                    continue
            elif len(cut_pos) == 2:
                pos1, pos2 = cut_pos
                _cinput[:,feat_id] = torch.max(torch.zeros(_cinput.size(0),device=device), 
                                                _cinput[:,feat_id]-pos1)
                _cinput[:,feat_id] = torch.min((pos2-pos1-offset[-1]-1)*torch.ones(_cinput.size(0),device=device), 
                                                    _cinput[:,feat_id])
                shard_output = _cinput[:,feat_id:feat_id+1] + offset
                assert torch.allclose(test_shard_output,shard_output)
                continue
            else:
                raise ValueError('input tensor and embeddings_per_feat do not match. Double check inputs.')

@pytest.mark.parametrize('world_size', [1,2,4,8])
@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_layer(world_size, device):
    run_func = partial(check_loadbalancemanager, world_size=world_size, device=device)
    run_func()

if __name__ == '__main__':
    test_layer(4, 'cpu')
