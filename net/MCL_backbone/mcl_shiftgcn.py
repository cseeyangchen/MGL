import torch
import torch.nn as nn
from net.utils.st_gcn.graph import Graph as STGraph
from net.utils.shiftgcn.graph import Graph
import torch.nn.functional as F
import math
from net.backbone.shiftgcn import TCN_GCN_unit


"""
    多视角融合方式3:MCL和Shift-GCN的结合
"""

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class ConvTemporalGraphical(nn.Module):
    """
    gcn--图卷积操作--空间特征提取
    """
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1,
                 t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0), stride=(t_stride, 1), dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)

        n, kc, t, v = x.size()
        # print("x:", x.size())
        # print("kernel_size:", self.kernel_size)
        # print("kc // self.kernel_size:", kc // self.kernel_size)
        # print("v:", v)
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        x = x.permute(0, 2, 1, 3)

        return x.contiguous(), A


class ConvLSTMCell(nn.Module):
    """
    对“每个时间步”的C*V特征进行一维卷积操作
    并得到每个时间块的h和c
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, device_id, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = kernel_size // 2

        # 一维卷积操作 -- 对C,V进行一维卷积操作 -- 针对单一特定视角
        self.conv1d = nn.Conv1d(in_channels=self.input_dim + self.hidden_dim,
                                out_channels=4 * self.hidden_dim * self.kernel_size,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)

        # 掩码
        self.Feature_Mask1 = nn.Parameter(torch.ones(1, 25, self.input_dim, requires_grad=True, device='cuda:'+str(device_id)),requires_grad=True)
        nn.init.constant_(self.Feature_Mask1, 0)
        self.Feature_Mask2 = nn.Parameter(torch.ones(1, 25, self.input_dim, requires_grad=True, device='cuda:'+str(device_id)),requires_grad=True)
        nn.init.constant_(self.Feature_Mask2, 0)
        self.Feature_Mask3 = nn.Parameter(torch.ones(1, 25, self.input_dim, requires_grad=True, device='cuda:'+str(device_id)),requires_grad=True)
        nn.init.constant_(self.Feature_Mask3, 0)

        self.fc = nn.Linear(in_features=self.input_dim+self.hidden_dim,
                            out_features=4*self.hidden_dim,
                            bias=self.bias)
        self.relu = nn.ReLU()

    def forward(self, input_tensor1, input_tensor2, input_tensor3, cur_state, A):
        # 注意力掩码操作
        n, c, v = input_tensor1.size()
        input_tensor1 = input_tensor1.permute(0, 2, 1).contiguous()
        input_tensor1 = input_tensor1*(torch.tanh(self.Feature_Mask1)+1)
        input_tensor2 = input_tensor2.permute(0, 2, 1).contiguous()
        input_tensor2 = input_tensor2 * (torch.tanh(self.Feature_Mask2) + 1)
        input_tensor3 = input_tensor3.permute(0, 2, 1).contiguous()
        input_tensor3 = input_tensor3 * (torch.tanh(self.Feature_Mask3) + 1)

        # 原始MCL：
        # input_tensor = input_tensor1+input_tensor2+input_tensor3    # n v c
        # 第二版MCL：借鉴MP取最大值，不进行加权
        input_tensor1 = input_tensor1.view(n, -1)
        input_tensor2 = input_tensor2.view(n, -1)
        input_tensor3 = input_tensor3.view(n, -1)
        input_tensor = torch.max(input_tensor1, input_tensor2)
        input_tensor = torch.max(input_tensor, input_tensor3)
        input_tensor = input_tensor.view(n, v, c)

        input_tensor = input_tensor.permute(0,2,1).contiguous()   # n c v

        # input_tensor1  维度为 n c v
        # n,c,v = input_tensor1.size()
        # input_tensor = torch.stack((input_tensor1, input_tensor2, input_tensor3), dim=2)   # 维度 n c 3 v
        # input_tensor = F.avg_pool2d(input_tensor,input_tensor.size()[2:])
        # input_tensor = input_tensor.view(n,c)

        h_cur, c_cur = cur_state
        # combined = self.conv1d(input_tensor)
        combined = torch.cat([input_tensor, h_cur],dim=1)
        # combined_fc = self.fc(combined)
        # combined_fc = self.relu(combined_fc)
        combined = self.conv1d(combined)

        # version2更改部分 -- gcn
        n, kc, v = combined.size()
        # print("combined:",combined.size())
        # print("kernel_size:",self.kernel_size)
        # print("kc // self.kernel_size:",kc // self.kernel_size)
        # print("v:",v)
        combined = combined.view(n, self.kernel_size, kc // self.kernel_size, v)
        combined = torch.einsum('nkcv, kvw->ncw',(combined, A))

        cc_i, cc_f, cc_o, cc_g = torch.split(combined, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, joint_size):
        return (torch.zeros(batch_size, self.hidden_dim, joint_size, device=self.fc.weight.device),
                torch.zeros(batch_size, self.hidden_dim, joint_size, device=self.fc.weight.device))


class MultiConvLSTM(nn.Module):
    """
    lstm,提取时序关系
    """
    def __init__(self,input_dim, hidden_dim, kernel_size, device_id, bias=True, view=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.view_num = view
        self.view_stream = ConvLSTMCell(input_dim,hidden_dim,kernel_size,device_id,bias)

    def forward(self, x, A, hidden_state=None):
        """
         Parameters
        ----------
        input_tensor: todo
            4-D Tensor either of shape (t, b, c, v) or (b, t, c, v)
        hidden_state: todo
            None. todo implement stateful
        """
        b, _, _, v = x[0].size()

        # Implement stateful GcnLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # hidden_state包括三个视角的初始state信息
            hidden_state = self.view_stream.init_hidden(batch_size=b,joint_size=v)

        cur_input = x
        seq_len = x[0].size(2)

        # 时间步
        output_inner = []
        last_state_list = []
        h, c = hidden_state
        for t in range(seq_len):
            h, c = self.view_stream(cur_input[0][:,:,t,:],cur_input[1][:,:,t,:],cur_input[2][:,:,t,:],[h,c],A)
            output_inner.append(h)
        last_state_list.append([h,c])

        layer_output = torch.stack(output_inner,dim=2)
        return layer_output, last_state_list


class GCN_LSTM_Fusion_Unit(nn.Module):
    """
    包含gcn和多支lstm融合--进行lstm间融合
    """
    def __init__(self,in_channels,out_channels,kernel_size,device_id,bias=True,view=3):
        super().__init__()
        self.view = view
        self.frame = 75
        self.joint_size = 25
        self.lstm = MultiConvLSTM(out_channels,out_channels,kernel_size[0],device_id,view=view)
        self.ln = nn.ModuleList([nn.LayerNorm([self.frame, out_channels, self.joint_size], elementwise_affine=False) for _ in range(view)])

    def forward(self, x, A):
        # gcn操作
        for view_index in range(self.view):
            x[view_index] = x[view_index].permute(0,2,1,3).contiguous()
            x[view_index] = self.ln[view_index](x[view_index].clone())
            x[view_index] = x[view_index].permute(0, 2, 1, 3).contiguous()
        x, _ = self.lstm(x, A, hidden_state=None)  # n c t v
        n,c,t,v = x.size()
        # x = x.permute(0,2,1).view(n,c,t,1)  # n c t v
        return x, A


class MCL_SHIFTGCN(nn.Module):
    def __init__(self, in_channels, num_class, device_id, graph_args_shiftgcn,
                 graph_args_stgcn, edge_importance_weighting):
        super().__init__()
        self.view_num = 3

        # load shiftgcn
        self.graph_shiftgcn = Graph(**graph_args_shiftgcn)
        A_shiftgcn = self.graph_shiftgcn.A
        self.data_bn = nn.BatchNorm1d(2 * in_channels * 25)

        self.shiftgcn = nn.ModuleList((
            TCN_GCN_unit(3, 64, A_shiftgcn, residual=False),
            TCN_GCN_unit(64, 64, A_shiftgcn),
            TCN_GCN_unit(64, 64, A_shiftgcn),
            TCN_GCN_unit(64, 64, A_shiftgcn),
            TCN_GCN_unit(64, 128, A_shiftgcn, stride=2),
            TCN_GCN_unit(128, 128, A_shiftgcn),
            TCN_GCN_unit(128, 128,  A_shiftgcn),
            TCN_GCN_unit(128, 256, A_shiftgcn, stride=2),
            TCN_GCN_unit(256, 256, A_shiftgcn),
            TCN_GCN_unit(256, 256, A_shiftgcn)
        ))
        self.predict_spe_stage = nn.ModuleList([nn.Linear(256, num_class) for _ in range(self.view_num)])
        for view_index in range(self.view_num):
            nn.init.normal_(self.predict_spe_stage[view_index].weight,0,math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

        # 第二阶段融合网络
        # load graph
        self.graph_stgcn = STGraph(**graph_args_stgcn)
        A_stgcn = torch.tensor(self.graph_stgcn.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_stgcn', A_stgcn)
        spatial_kernel_size_stage2 = A_stgcn.size(0)
        temporal_kernel_size_stage2 = 1  # 只能为奇数
        kernel_size_stage2 = (spatial_kernel_size_stage2, temporal_kernel_size_stage2)
        self.gcn_lstm_fusion_networks = GCN_LSTM_Fusion_Unit(256, 256, kernel_size_stage2, device_id, view=self.view_num)

        # 第三阶段 -- 分类器
        self.late_fusion_stage = nn.Conv2d(256, num_class, kernel_size=1)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance_stage2 = nn.Parameter(torch.ones(self.A_stgcn.size(), device='cuda:'+str(device_id)))

    def data_process(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        return x

    def forward(self, x1, x2, x3):
        N, C, T, V, M = x1.size()
        # print("step1: ", torch.cuda.memory_allocated() / 1024 / 1024)
        # 预处理阶段--data_bns--PreLayer
        x1 = self.data_process(x1)
        x2 = self.data_process(x2)
        x3 = self.data_process(x3)
        x = [x1, x2, x3]
        # print("step2: ", torch.cuda.memory_allocated() / 1024 / 1024)
        # stage1--第一阶段网络--特定单视角学习
        for view_index in range(self.view_num):
            # print("step3."+str(view_index)+": ", torch.cuda.memory_allocated() / 1024 / 1024)
            for shiftgcn_layer in self.shiftgcn:
                x[view_index] = shiftgcn_layer(x[view_index])
        # prediction
        # print("step3: ", torch.cuda.memory_allocated() / 1024 / 1024)
        output1 = x.copy()
        for view_index in range(self.view_num):
            c_new = output1[view_index].size(1)
            output1[view_index] = output1[view_index].view(N, M, c_new, -1)
            output1[view_index] = output1[view_index].mean(3).mean(1)
            output1[view_index] = self.predict_spe_stage[view_index](output1[view_index])

        # print("step4: ", torch.cuda.memory_allocated() / 1024 / 1024)
        # stage2--第二阶段融合网络--多视角联合学习
        x, _ = self.gcn_lstm_fusion_networks(x, self.A_stgcn * self.edge_importance_stage2)
        # print("step5: ", torch.cuda.memory_allocated() / 1024 / 1024)
        # stage3--融合特征，而不是标签--在时间维度t上
        output3 = F.avg_pool2d(x, x.size()[2:])
        output3 = output3.view(N, M, -1, 1, 1).mean(dim=1)
        output3 = self.late_fusion_stage(output3)
        output3 = output3.view(output3.size(0), -1)
        # print("step6: ", torch.cuda.memory_allocated() / 1024 / 1024)

        return output1, output3


if __name__ == "__main__":
    model = MCL_SHIFTGCN(in_channels=3,num_class=60,device_id=0,edge_importance_weighting=True,
                     graph_args_shiftgcn={'labeling_mode': 'spatial'},
                     graph_args_stgcn={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
    x1 = torch.rand((2,3,300,25,2))
    x2 = torch.rand((2, 3, 300, 25, 2))
    x3 = torch.rand((2, 3, 300, 25, 2))
    o1, o2 = model(x1,x2,x3)