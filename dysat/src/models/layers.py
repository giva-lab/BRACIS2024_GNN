# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2024/08/27 21:10:00
@Author  :   Waqar Hassan 
@Contact :   waqar_comsat@yahoo.com
USP, Sao Paulo, Brazil
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import softmax
from torch_scatter import scatter
import pdb

import copy


class StructuralAttentionLayer(nn.Module):
    def __init__(self, 
                input_dim, 
                output_dim, 
                n_heads, 
                attn_drop, 
                ffd_drop,
                residual):
        super(StructuralAttentionLayer, self).__init__()
        self.out_dim = output_dim // n_heads
        self.n_heads = n_heads
        self.act = nn.ELU()

        self.lin = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)
        self.att_l = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))
        self.att_r = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.attn_drop = nn.Dropout(attn_drop)
        self.ffd_drop = nn.Dropout(ffd_drop)

        self.residual = residual
        if self.residual:
            self.lin_residual = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)

        self.xavier_init()

    def forward(self, graph):
        graph = copy.deepcopy(graph)
        edge_index = graph.edge_index
        edge_weight = graph.edge_weight.reshape(-1, 1)  # Converting to a column vector
        H, C = self.n_heads, self.out_dim
        x = self.lin(graph.x).view(-1, H, C) # [N, heads, out_dim] # Reshaping
        
        # attention
        alpha_l = (x * self.att_l).sum(dim=-1).squeeze() # [N, heads]
        alpha_r = (x * self.att_r).sum(dim=-1).squeeze()
        alpha_l = alpha_l[edge_index[0]] # [num_edges, heads]  # Extracting source nodes weights
        alpha_r = alpha_r[edge_index[1]]    ## Extracting Target nodes weights
        alpha = alpha_r + alpha_l
        alpha = edge_weight * alpha
        alpha = self.leaky_relu(alpha)      #Overall, using Leaky ReLU before applying softmax can help enhance the learning dynamics and robustness of the neural network model, leading to potentially improved performance, especially in deeper architectures or when dealing with challenging datasets.
        coefficients = softmax(alpha, edge_index[1]) # [num_edges, heads] #converts a vector of arbitrary real values into a vector of probabilities, where each element is in the range (0, 1) and the sum of all elements is 1. 
                                                     # attention scores are normalized across the destination nodes, producing a valid probability distribution.
        # dropout
        if self.training:
            coefficients = self.attn_drop(coefficients)
            x = self.ffd_drop(x)
        x_j = x[edge_index[0]]  # [num_edges, heads, out_dim] # Extracting node information of only source nodes in edges

        # output
        out = self.act(scatter(x_j * coefficients[:, :, None], edge_index[1], dim=0, reduce="sum")) # Scatter() is used to aggregate information from neighboring nodes
        out = out.reshape(-1, self.n_heads*self.out_dim) #[num_nodes, output_dim]
        if self.residual:
            out = out + self.lin_residual(graph.x)
        graph.x = out
        return graph

    def xavier_init(self):
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

        
class TemporalAttentionLayer(nn.Module):
    def __init__(self, 
                input_dim, 
                n_heads, 
                num_time_steps, 
                attn_drop, 
                residual):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # ff
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout 
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()


    def forward(self, inputs):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        # 1: Add position embeddings to input
        position_inputs = torch.arange(0,self.num_time_steps - 1).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(inputs.device)
        temporal_inputs = inputs + self.position_embeddings[position_inputs] # [N, T, F] adding position embeddings to the input sequences

        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2],[0])) # [N, T, F]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2],[0])) # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2],[0])) # [N, T, F]

        # 3: Split, concat and scale.
        split_size = int(q.shape[-1]/self.n_heads)
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h] This operation stacks the queries for each attention head on top of each other, resulting in a tensor q_ with shape [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        
        outputs = torch.matmul(q_, k_.permute(0,2,1)) # [hN, T, T]
        outputs = outputs / (self.num_time_steps ** 0.5)  #This operation ensures that the attention scores are adjusted appropriately for the number of time steps (T) to prevent the magnitude of the dot products from becoming too large or too small, thereby facilitating stable training.
        
        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = torch.ones_like(outputs[0])  #This creates a tensor of ones with the same shape as the first element of the outputs tensor. It is used to generate the diagonal values of the lower triangular matrix.
        tril = torch.tril(diag_val)
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1) # [h*N, T, T]
        padding = torch.ones_like(masks) * (-2**32+1)
        outputs = torch.where(masks==0, padding, outputs)
        outputs = F.softmax(outputs, dim=2)
        self.attn_wts_all = outputs # [h*N, T, T]
                
        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0]/self.n_heads), dim=0), dim=2) # [N, T, F]
        
        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = outputs + temporal_inputs
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs


    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)
