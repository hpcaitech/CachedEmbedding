import math
from typing import List

import torch
from colossalai.registry import GRADIENT_HANDLER
from colossalai.engine.gradient_handler import BaseGradientHandler


@GRADIENT_HANDLER.register_module
class CowclipGradientHandler(BaseGradientHandler):

    def handle_gradient(self, embed_dim: List, clip: int = 1.0, bound: int = 1.0):
        gradients = [] 
        trainable_vars = list(self._model.named_parameters())

        for param in trainable_vars:
            gradients.append(param[1].grad)

        embed_index = [
            i for i, x in enumerate(trainable_vars) if "embedding" in x[0]
        ]
        dense_index = [i for i in range(
            len(trainable_vars)) if i not in embed_index]
        
        embed_vars = [trainable_vars[i] for i in embed_index]
        _dense_vars = [trainable_vars[i] for i in dense_index]
        embed_gradients = [gradients[i] for i in embed_index]
        dense_gradients = [gradients[i] for i in dense_index]

        # CowClip
        if clip > 0:
            lower_bound = clip * math.sqrt(embed_dim) * bound
            embed_gradients_clipped = []
            for w, g in zip(embed_vars, embed_gradients):
                
                if 'embedding' not in w[0]:
                    embed_gradients_clipped.append(g)
                    continue
                
                g_clipped = self._compute_cow_clip(w[1], torch.reshape(g, w[1].size()), ratio=clip,
                                            ids=None, cnts=None, min_w=lower_bound)

                embed_gradients_clipped.append(g_clipped)

            embed_gradients = embed_gradients_clipped

        gradients = embed_gradients + dense_gradients

        with torch.no_grad():
            index = embed_index+dense_index 
            for i,j in enumerate(index):
                trainable_vars[j][1].grad = gradients[i]

        super(self).handle_gradient()

    @staticmethod
    def _compute_cow_clip(w, g, ratio=1, ids=None, cnts=None, min_w=0.03, const=False):

        if isinstance(g, dict):
            values = torch.tensor(g.values())
            clipnorm = torch.norm(torch.gather(w, g.indices), axis=-1)
        else:
            values = g
            if const:
                clipnorm = torch.tensor([min_w] * g.shape[0], requires_grad=False)
            else:
                clipnorm = torch.linalg.norm(w, axis=-1)

                clipnorm = torch.max(clipnorm, clipnorm.new_tensor([min_w] * clipnorm.size(0), requires_grad=False))

        clip_t = ratio * clipnorm
        l2sum_row = torch.sum(values * values, axis=-1)
        pred = l2sum_row > 0
        l2sum_row_safe = torch.where(pred, l2sum_row, torch.ones_like(l2sum_row))
        l2norm_row = torch.sqrt(l2sum_row_safe)

        intermediate = values * clip_t.unsqueeze(-1)

        g_clip = intermediate / torch.maximum(l2norm_row, clip_t).unsqueeze(-1)

        return g_clip
    
