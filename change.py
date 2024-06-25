import torch

state_dict = torch.load("model/MVF_Spatial_Temporal_Occ/MCL_model/Epoch32_MCL_model.pt")
torch.save(state_dict, "Epoch32_MCL_model.pt", _use_new_zipfile_serialization=False)
