from pdb import set_trace as keyboard

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

#########################################################################
#                        CVD_MLP_v1()
#########################################################################
class CVD_MLP_v1(nn.Module):
    """
    A shallow MLP to be used for the MoG dataset

    Malinin's thesis, p. 45, caption of Fig. 3.8
    The DNNs had 2 layers of 100 ReLU units.    

    """

    def __init__(self, n_hidden=2, hidden_sz=100, p_dropout=0.1):
        super(CVD_MLP_v1, self).__init__()

        self.n_in = 456 # toy/mog.py defines a 2D dataset
        self.n_hidden = n_hidden # number of hidden layers (e.g., 1 or 2), default=2
        self.hidden_sz = hidden_sz # number of elements in each hidden layer (e.g., 50, 100), default=100
        self.n_out = 12 # toy/mog.py defines a 3-class dataset
        self.p_dropout = p_dropout

        mdl_spec = OrderedDict([('fc1', nn.Linear(self.n_in, self.hidden_sz)),
                                ('dropout1', nn.Dropout(p=self.p_dropout)),
                                ('relu1', nn.ReLU())])

        for i in range(1, n_hidden):
            mdl_spec.update(OrderedDict([(f"fc{i+1}", nn.Linear(self.hidden_sz, self.hidden_sz)),
                                         (f"dropout{i+1}", nn.Dropout(p=self.p_dropout)),
                                         (f"relu{i+1}", nn.ReLU())]))
                                        

        """
        :NOTE: following Malinin's my_vgg.MyVGG.classifier
        the final layer is nn.Linear i.e., logits (rather than softmax)
        """
        mdl_spec.update(OrderedDict([(f'fc{n_hidden+1}',nn.Linear(self.hidden_sz, self.n_out))]))
                            
        self.m_cls = nn.Sequential(mdl_spec)
        

    def forward(self, x):
        output = self.m_cls(x)
        return output


def cvd_mlp(n_hidden, hidden_sz, p_dropout):
    return CVD_MLP_v1(n_hidden, hidden_sz, p_dropout)
