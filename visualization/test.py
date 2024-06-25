#torch_version==1.x
import torch
from net.MultiView import MCL_model_v2 as MCL_model
from collections import OrderedDict
checkpoint = 'ntu60_Epoch31_MCL_model_v2.pt'

model =MCL_model.MCL(in_channels=3,num_class=60,device_id=0,edge_importance_weighting=True,
                    graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
weights = torch.load(checkpoint)
weights = OrderedDict([[k.split('module.')[-1],v.cpu()] for k, v in weights.items()])
model.load_state_dict(weights)
model.eval()

torch.save(model.state_dict(), "ntu60_MCL_model_transfer.pth", _use_new_zipfile_serialization=False)
