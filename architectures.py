import numpy as np

import torch
import torch.nn as nn

from torch_geometric.nn import EdgeConv
from torch_geometric.nn import global_mean_pool

class LundNet(nn.Module):

    def __init__(
            self,
            conv_params,
            fc_params,
            input_dims = 3,
            use_fusion = True,
            num_classes = 1,
            add_fractions_to_lund = False,
            ):
        super(LundNet, self).__init__()

        self.conv_params = conv_params
        self.fc_params = fc_params
        self.use_fusion = use_fusion
        self.num_classes = num_classes
        self.add_fractions_to_lund = add_fractions_to_lund
        self.input_dims = input_dims

        if self.add_fractions_to_lund:
            self.input_dims += 4

        #batch normalization features
        self.bn_fts = nn.BatchNorm1d(self.input_dims)

        self.edge_convs = nn.ModuleList()
        for idx, channels in enumerate(conv_params):
            in_feat = self.input_dims if idx == 0 else conv_params[idx - 1][-1]
            out_feat = conv_params[idx][1] 
            nns = nn.Sequential(nn.Linear(2 *in_feat, out_feat),
                        nn.BatchNorm1d(out_feat),
                        #added dropout
                        nn.Dropout(0.6),
                        nn.ReLU(),
                        nn.Linear(out_feat, out_feat),
                        nn.ReLU(),
                        nn.BatchNorm1d(out_feat),
                        #added dropout
                        nn.Dropout(0.6))
            self.edge_convs.append(EdgeConv(nn=nns, aggr='mean'))

        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(nn.Linear(in_chn, out_chn), nn.ReLU(), nn.BatchNorm1d(out_chn))

        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][-1]
            else:
                in_chn = fc_params[idx - 1][0]
            #fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU() ,nn.Dropout(drop_rate)))
            fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.BatchNorm1d(channels), nn.Dropout(drop_rate)))
        #fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        fcs.append(nn.Sequential(nn.Linear(fc_params[-1][0], num_classes),nn.BatchNorm1d(num_classes)))
        self.fc = nn.Sequential(*fcs)

    def forward(self, data):
    
        x, edge_index = data.x, data.edge_index
        #batch normalization features
        x = self.bn_fts(x)
        
        outputs = []
        for idx, conv in enumerate(self.edge_convs):
            x = conv(x, edge_index)
            if self.use_fusion:
                outputs.append(x)
        #fuse outputs from all EdgeConv layers
        if self.use_fusion:
            x = self.fusion_block(torch.cat(outputs, dim=1))

        #Global average pooling
        x = global_mean_pool(x, batch = data.batch) 
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x