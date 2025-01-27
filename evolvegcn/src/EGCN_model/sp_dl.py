import utils as u
import os
import torch
import pickle
import time
import numpy as np

# FULLY MODIFIED FROM ORIGINAL

class Sao_Paulo_DataSet():
    def __init__(self,args):
        args.sp_args = u.Namespace(args.sp_args)
        
        nodes_labels_times_file = open(args.sp_args.nodes_labels_times_file, 'rb')
        edges_file = open(args.sp_args.edges_file, 'rb')
        nodes_file = open(args.sp_args.nodes_file, 'rb')
        nodes_feats_file = open(args.sp_args.nodes_feats_file, 'rb')
        
        self.nodes_labels_times = torch.from_numpy((pickle.load(nodes_labels_times_file)).astype(int))
        self.edges = pickle.load(edges_file)
        self.nodes = torch.from_numpy((pickle.load(nodes_file)).astype(float))
        self.nodes_feats = torch.from_numpy((pickle.load(nodes_feats_file)).astype(float))
        
        self.num_nodes = len(self.nodes)
        self.feats_per_node = self.nodes.shape[1]-1
        
        self.edges = self.replicate_times(args)

        
    def replicate_times(self, args):
        cop = self.edges.copy()
        aux = cop['idx'].copy()
        for i in range(self.nodes_labels_times[-1][-1]):
            idxs = [(i+1)*cop['idx'].shape[0], (i+2)*cop['idx'].shape[0]]
            
            aux = np.concatenate((aux, cop['idx']))
            aux[idxs[0]:idxs[1], -1] = [i+1]
        
  
        cop['idx'] = torch.from_numpy(aux.astype(int)) 
        cop['vals'] = torch.ones(cop['idx'].shape[0])
        
        self.max_time = cop['idx'][:,2].max()
        self.min_time = cop['idx'][:,2].min()
        
        return cop
        