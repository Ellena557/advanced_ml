import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm


class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, concat_number_of_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()
        self.title_emb = nn.Embedding(n_tokens, embedding_dim=hid_size)
        #self.description_emb = nn.Embedding(n_tokens, embedding_dim=hid_size)
        self.category_dense = nn.Linear(in_features=n_cat_features, out_features=hid_size)
        
        self.full_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        self.conv1 = nn.Conv1d(in_channels = hid_size, out_channels = hid_size, kernel_size = 2)
        self.conv2 = nn.Conv1d(in_channels = hid_size, out_channels = hid_size, kernel_size = 2)
        self.relu = nn.ReLU()
        #self.pool = nn.AdaptiveMaxPool1d(1)
        self.pool1 = nn.AdaptiveAvgPool1d(1)
        self.pool2 = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(0.25)

        self.category_out = nn.Linear(in_features=n_cat_features, out_features=hid_size)

        # Example for the final layers (after the concatenation)
        self.inter_dense = nn.Linear(in_features=concat_number_of_features, out_features=hid_size*2)
        self.final_dense = nn.Linear(in_features=hid_size*2, out_features=1)

        

    def forward(self, whole_input):
        input1, input2, input3 = whole_input
        title_beg = self.title_emb(input1).permute((0, 2, 1))
        # <YOUR CODE HERE>
        title = self.conv1(title_beg)
        title = self.pool1(title)
        
        full_beg = self.full_emb(input2).permute((0, 2, 1))
        full = self.conv2(full_beg)
        full = self.pool2(full)
        
        category = self.category_out(input3)
        category = self.dropout(category)
        
        concatenated = torch.cat(
            [
            title.view(title.size(0), -1),
            full.view(full.size(0), -1),
            category.view(category.size(0), -1)
            ],
            dim=1)
        
        # <YOUR CODE HERE>
        out = self.inter_dense(concatenated)
        out = self.relu(out)
        out = self.final_dense(out)
        
        return out
