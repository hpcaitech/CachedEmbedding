from functools import partial
from pyparsing import Word

import pytest
import torch 
import numpy as np

from recsys.modules.embeddings import LoadBalanceManager

from common import  NUM_EMBEDDINGS_PER_FEATURE, BATCH_SIZE, check_equal


def check_loadbalancemanager(be_fair, world_size, device):
    lbmgr = LoadBalanceManager(embeddings_per_feat=NUM_EMBEDDINGS_PER_FEATURE,
                               num_groups=world_size,
                               be_fair=be_fair,
                               device=device)
    
    rand_input = []
    
    for i in range(len(NUM_EMBEDDINGS_PER_FEATURE)):
        rand_input.append(torch.randint(0,NUM_EMBEDDINGS_PER_FEATURE[i],
                                               size=(BATCH_SIZE,)).unsqueeze(1))
        
    rand_input_tensor = torch.cat(rand_input,dim=1)
    
    assert rand_input_tensor.shape == (BATCH_SIZE, len(NUM_EMBEDDINGS_PER_FEATURE))
    
    if be_fair:
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
        
        # Perform tensor sharding
        for rank in range(world_size):
            offsets = offsets[rank]
            shard_output = torch.empty(size=(BATCH_SIZE,len(offsets)))
            _cinput = rand_input_tensor.clone()
            feats = list(cuts.keys())
            if rank == 0:
                feat_id = feats[0]
                cut_pos = cuts[feat_id][0]
                _cinput[:,feat_id] = torch.min(cut_pos*torch.ones(_cinput.size(0)), _cinput[:,feat_id])
                shard_output = _cinput[:,:feat_id+1] + offsets
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
                        _cinput[:,feat_id] = torch.max(torch.zeros(_cinput.size(0)), _cinput[:,feat_id]-cut_pos)
                        shard_output = _cinput[:,feat_id:] + offsets
                    else:
                        assert next_feat_id is not None
                        _cinput[:,feat_id] = torch.max(torch.zeros(_cinput.size(0)), 
                                                    _cinput[:,feat_id]-cut_pos)
                        _cinput[:,next_feat_id] = torch.min(cut_pos*torch.ones(_cinput.size(0)), 
                                                            _cinput[:,next_feat_id])
                        shard_output = _cinput[:,:next_feat_id+1] + offsets
                elif len(cut_pos) == 2:
                    pos1, pos2 = cut_pos
                    _cinput[:,feat_id] = torch.max(torch.zeros(_cinput.size(0)), 
                                                    _cinput[:,feat_id]-pos1)
                    _cinput[:,feat_id] = torch.min(pos2*torch.ones(_cinput.size(0)), 
                                                        _cinput[:,feat_id])
                    shard_output = _cinput[:,feat_id] + offsets
                else:
                    raise ValueError('input tensor and embeddings_per_feat do not match. Double check inputs.')
                
            assert check_equal(lbmgr.shard_tensor(rand_input_tensor,rank).detach(), shard_output.detach())
    else:
        dim_indices = np.array(range(len(NUM_EMBEDDINGS_PER_FEATURE)))
        np.random.shuffle(dim_indices)
        chunk_size = len(NUM_EMBEDDINGS_PER_FEATURE) // world_size
        groups = []

        for i in range(world_size):
            if i == world_size-1:
                groups.append(dim_indices[i*chunk_size:])
                break
            groups.append(dim_indices[i*chunk_size:(i+1)*chunk_size])
            
        offsets = [torch.tensor((0,*np.cumsum(np.array( \
                            NUM_EMBEDDINGS_PER_FEATURE, dtype=np.long)[group])[:-1]), device=device)
                        for group in groups]
        
        group = groups[rank]
        check_equal(lbmgr.shard_tensor(rand_input_tensor,rank), rand_input_tensor[:, group] + offsets)


@pytest.mark.parametrize('be_fair', [True,False])
@pytest.mark.parametrize('world_size', [1,2,4,8])
@pytest.mark.parametrize('device', ['cpu','cuda:0'])
def test_layer(be_fair, world_size, device):
    run_func = partial(check_loadbalancemanager, be_fair=be_fair, world_size=world_size, device=device)
    run_func()

if __name__ == '__main__':
    test_layer(True,4,'cpu')
