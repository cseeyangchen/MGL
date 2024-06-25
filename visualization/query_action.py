import numpy as np
result_dict = np.load('top5_occlusion.npy',allow_pickle=True).item()
# print(result_dict)
# 查询动作
for k, v in result_dict.items():
    action_name = k.split('-')[-1]
    if action_name == "writing":
        print(k)
        print(v)


