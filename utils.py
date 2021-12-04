# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:46:47 2021

@author: James
"""

import torch
from torch import nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy


#LabelSmoothingLoss具有一定正则化功能，能够防止过拟合，因为其函数形式为对勾型，当p过大时loss反而会变大。
class LabelSmoothingLoss(nn.Module):
    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.0):
        super(LabelSmoothingLoss,self).__init__()
        """
        size：class的个数
        padding_idx：置零列的索引
        smoothing：平滑量，对应的是confidence，指的是当p=confidence时就可以认为是真的了，不用到1.
        """
        self.loss = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        assert x.shape[1] == self.size
        true_dist = x.clone()
        #除了0和confidence以外的元素相加=smoothing，所以平均分配smoothing。
        true_dist.fill_(self.smoothing / (self.size - 2))
        #在target位置填入confidence值
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        #把第0列置零，使得整体相加概率为1
        true_dist[:, self.padding_idx] = 0
        
        #如果置零的列的某行恰好有confidence则把这一行全都置零
        mask = torch.nonzero(target == self.padding_idx, as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.loss(x, true_dist.detach())


def _test_label_smoothing():
    smooth_loss = LabelSmoothingLoss(5, 0, 0.4)
    predict = torch.tensor([[0, 0.2, 0.7, 0.1, 0],
                            [0, 0.2, 0.7, 0.1, 0],
                            [0, 0.2, 0.7, 0.1, 0]], dtype=torch.float)
    _ = smooth_loss(predict.log(),
                    torch.tensor([2, 1, 0], dtype=torch.long))

    # Show the target distributions expected by the system.
    plt.imshow(smooth_loss.true_dist)
    plt.show()

    smooth_loss = LabelSmoothingLoss(5, 0, 0.1)

    def loss_sample(x):
        d = x + 3 * 1
        predict2 = torch.tensor([[0, x / d, 1 / d, 1 / d, 1 / d],], dtype=torch.float)
        # print(predict)
        return smooth_loss(predict2.log(),
                           torch.tensor([1], dtype=torch.long)).item()

    plt.plot(np.arange(1, 100), [loss_sample(x) for x in range(1, 100)])
    plt.show()


# _test_label_smoothing()


def subsequent_mask(seq_len):
    """
    ## Subsequent mask to mask out data from future (subsequent) time steps
    因为是下三角矩阵，所以每行表示不同时间之间的注意力。
    """
    mask = torch.tril(torch.ones(seq_len, seq_len)).to(torch.bool).unsqueeze(-1)
    
    return mask


def clone_module_list(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



class CrossEntropyLoss(nn.Module):
    """
    ### Cross entropy loss
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        return self.loss(outputs.view(-1, outputs.shape[-1]), targets.view(-1))

