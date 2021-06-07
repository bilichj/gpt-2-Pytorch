from __future__ import annotations
from typing import List, Optional
import torch


class Node:
    def __init__(self, parent, *args, **kwargs):
        self.parent = parent
    
    @property
    def anscestors(self):
        anscestor = self
        while anscestor is not None:
            yield anscestor
            anscestor = anscestor.parent

class GPT2Node(Node):
    def __init__(self,
                parent: Optional[GPT2Node]=None,
                token_id: Optional[int]=None,
                presents: Optional[torch.Tensor]=None,
                logits: Optional[torch.Tensor]=None,
                *args, **kwargs):
        
        super().__init__(parent, *args, **kwargs)

        self.token_id = token_id
        self.presents = presents
        self.logits = logits
    
    def __len__(self):
        return len(list(self.anscestors))
    
    @property
    def presents(self):
        if self.parent is None:
            return [torch.zeros([2, 1, 12, 0, 64], 
                               dtype=torch.float32) for _ in range(12)]
        return self._presents

    @presents.setter
    def presents(self, value):
        self._presents = value

    @property
    def tokens(self) -> torch.Tensor:
        return torch.tensor([
            list(n.token_id for n in self.anscestors)
            ], dtype=torch.long)

    @property
    def past(self):
        if self.parent is None:
            return [None for _ in range(12)]
        return [torch.cat(X, dim=-2) for X in zip(*(n.presents for n in self.anscestors))]

    @property
    def root(self):
        if self.parent is None: return self
        return self.parent.root
    
    @property
    def cumulative_logit(self):
        cl = self.logit
        
        if self.parent is not None:
            cl += self.parent.logit

        return cl
        
    @property
    def logit(self):
        return self.logit