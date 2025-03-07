import torch
from net.MultiView import MCL_model


# 加载模型和权重
model = MCL_model.MCL(in_channels=3,num_class=60,device_id=0,edge_importance_weighting=True,
                    graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
model_dict = model.state_dict()
path = "model/st_gcn_pretrained/st_gcn.ntu-xsub.pt"
pretrained_weights1 = torch.load(path)
pretrained_weights1_layer = [k for k, v in pretrained_weights1.items()]
pretrained_weights = {k:v for k, v in pretrained_weights1.items() if k in model_dict}
model_dict.update(pretrained_weights)
model.load_state_dict(model_dict)
# 冻结权重
for name, param in model.named_parameters():
    if name in pretrained_weights1_layer:
        param.requires_grad=False
















                  