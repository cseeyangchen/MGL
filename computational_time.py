import torch
from thop import profile

# 加载多视角融合模型
from net.flops import CF_model
from net.flops import CS_model
from net.flops import MP_model
from net.flops import AP_model
from net.flops import LSTM_model
from net.flops import MCL_model

# 加载模型
def cal(modelName):
    if modelName == 'CF_model':
        model = CF_model.ConcatenateFeature(in_channels=3,num_class=60,edge_importance_weighting=True,
                    graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
    elif modelName == 'CS_model':
        model = CS_model.ConcatenateScore(in_channels=3,num_class=60,edge_importance_weighting=True,
                    graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
    # elif modelName == 'CNNS_model':
    #     model = CNNS_model.ConvolutionalScore(in_channels=3,num_class=60,edge_importance_weighting=True,
    #                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
    elif modelName == 'MP_model':
        model = MP_model.MaxPoolingFeature(in_channels=3,num_class=60,edge_importance_weighting=True,
                    graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
    elif modelName == 'AP_model':
        model = AP_model.AveragePoolingFeature(in_channels=3,num_class=60,edge_importance_weighting=True,
                    graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
    elif modelName == 'LSTM_model':
        model = LSTM_model.LstmFeature(in_channels=3,num_class=60,edge_importance_weighting=True,
                    graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
    elif modelName == 'MCL_model':
        model = MCL_model.MCL(in_channels=3,num_class=60,device_id=0,edge_importance_weighting=True,
                    graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
    else:
        print("We only implemented CF, CS, CNNS, MP, AP, LSTM and MCL models.")
        raise NotImplementedError
    return model


if __name__ == "__main__":
    model = cal('MCL_model')
    device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
    model.to(device)
    x1 = torch.rand((2, 256, 75, 25))
    x2 = torch.rand((2, 256, 75, 25))
    x3 = torch.rand((2, 256, 75, 25))
    flops, para = profile(model, inputs=(x1, x2, x3))
    print("%.2fM" % (flops / 1e6), "%.2fM" % (para / 1e6))




