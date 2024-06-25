import torch
from torch import nn

from .attentions import Attention_Layer
from .layers import Spatial_Graph_Layer, Temporal_Basic_Layer

from net.utils.efficientgcn import utils as U
from .activations import Swish
from net.utils.efficientgcn.graphs import Graph

class EfficientGCN(nn.Module):
    def __init__(self, data_shape, block_args, fusion_stage, stem_channel, **kwargs):
        super(EfficientGCN, self).__init__()

        num_channel, _, _, _ = data_shape

        # input branches
        self.input_branches = EfficientGCN_Blocks(
            init_channel = stem_channel,
            block_args = block_args[:fusion_stage],
            input_channel = num_channel,
            **kwargs)

        # main stream
        last_channel = stem_channel if fusion_stage == 0 else block_args[fusion_stage-1][0]
        self.main_stream = EfficientGCN_Blocks(
            init_channel =  last_channel,
            block_args = block_args[fusion_stage:],
            **kwargs
        )

        # output
        last_channel = block_args[-1][0]

        # init parameters
        init_param(self.modules())

    def forward(self, x):

        N, C, T, V, M = x.size()
        x = x.permute(0,4,1,2,3).contiguous().view(N*M, C, T, V)

        # input branches
        x = self.input_branches(x)

        # main stream
        x = self.main_stream(x)

        # output
        _, C, T, V = x.size()
        feature = x.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)

        return feature



class EfficientGCN_Blocks(nn.Sequential):
    def __init__(self, init_channel, block_args, layer_type, kernel_size, input_channel=0, **kwargs):
        super(EfficientGCN_Blocks, self).__init__()

        temporal_window_size, max_graph_distance = kernel_size

        if input_channel > 0:  # if the blocks in the input branches
            self.add_module('init_bn', nn.BatchNorm2d(input_channel))
            self.add_module('stem_scn', Spatial_Graph_Layer(input_channel, init_channel, max_graph_distance, **kwargs))
            self.add_module('stem_tcn', Temporal_Basic_Layer(init_channel, temporal_window_size, **kwargs))

        last_channel = init_channel
        temporal_layer = U.import_class(f'net.backbone.efficientgcn.layers.Temporal_{layer_type}_Layer')

        for i, [channel, stride, depth] in enumerate(block_args):
            self.add_module(f'block-{i}_scn', Spatial_Graph_Layer(last_channel, channel, max_graph_distance, **kwargs))
            for j in range(depth):
                s = stride if j == 0 else 1
                self.add_module(f'block-{i}_tcn-{j}', temporal_layer(channel, temporal_window_size, stride=s, **kwargs))
            self.add_module(f'block-{i}_att', Attention_Layer(channel, **kwargs))
            last_channel = channel


def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class EfficientGCN_Classifier(nn.Sequential):
    def __init__(self, curr_channel, num_class, drop_prob, **kwargs):
        super(EfficientGCN_Classifier, self).__init__()

        self.add_module('gap', nn.AdaptiveAvgPool3d(1))
        self.add_module('dropout', nn.Dropout(drop_prob, inplace=True))
        self.add_module('fc', nn.Conv3d(curr_channel, num_class, kernel_size=1))


class Model(nn.Module):
    def __init__(self, data_shape, block_args, fusion_stage, stem_channel, **kwargs):
        super(Model, self).__init__()

        num_channel, _, _, _ = data_shape

        # input branches
        self.input_branches = EfficientGCN_Blocks(
            init_channel = stem_channel,
            block_args = block_args[:fusion_stage],
            input_channel = num_channel,
            **kwargs)

        # main stream
        last_channel = stem_channel if fusion_stage == 0 else block_args[fusion_stage-1][0]
        self.main_stream = EfficientGCN_Blocks(
            init_channel =  last_channel,
            block_args = block_args[fusion_stage:],
            **kwargs
        )

        # output
        last_channel = block_args[-1][0]
        self.classifier = EfficientGCN_Classifier(last_channel, **kwargs)

        # init parameters
        init_param(self.modules())

    def forward(self, x):

        N, C, T, V, M = x.size()
        x = x.permute(0,4,1,2,3).contiguous().view(N*M, C, T, V)

        # input branches
        x = self.input_branches(x)

        # main stream
        x = self.main_stream(x)

        # output
        _, C, T, V = x.size()
        feature = x.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)
        out = self.classifier(feature).view(N, -1)

        return out

if __name__ == "__main__":
    block_args = U.rescale_block(block_args=[[48,1,0.5],[24,1,0.5],[64,2,1],[128,2,1]],scale_args=[1.2,1.35],scale_factor=0)
    model = EfficientGCN(data_shape=[3, 300, 25, 2],block_args=block_args,stem_channel=64,fusion_stage=2,
                         model_args={"act":Swish(inplace=True), "layer_type": "Sep","att_type": "stja",
                                     'num_class':60, "kernel_size": [5,2],
                                     'A': torch.Tensor(Graph("ntu-xsub").A),'parts': Graph("ntu-xsub").parts,
                                     "expand_ratio": 2,"reduct_ratio": 4,"bias": True,"edge": True})


