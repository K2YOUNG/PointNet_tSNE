import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

import importlib
import sys, os
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.abspath(os.path.join(CUR_DIR, os.pardir))
sys.path.append(BASE_PATH)
sys.path.append(CUR_DIR)

from models.pointnet_sem_seg import get_model

class PointNetFeatureExtractor(nn.Module):
    def __init__(self, n_kp, radius, n_samples, device):
        super(PointNetFeatureExtractor, self).__init__()
        self.kp = n_kp
        self.radius = radius
        self.n_samples = n_samples
        
        # Load pretrained PointNet
        model = get_model(13)
        checkpoint = torch.load(os.path.join(BASE_PATH, "log/sem_seg/pointnet_sem_seg/checkpoints/best_model.pth"), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Feature Extractor (PointNet Encoder)
        self.encoder = model.feat
    
    def forward(self, points):
        '''
        Input
            points : Input Point Cloud Data (size : [B, D, N])
            ( B : batch_size, D: coordinates & features, N : number of points)
        
        Returns
            embed : embeddings for each group
        '''

        # Extract Feature Embeddings
        x, trans, trans_feat = self.encoder(points)
        embed = x[:,:1024,0:1]

        return embed