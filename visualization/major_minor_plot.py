import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
import sys
sys.path.append("..")
from feeder.feeder_fusion import Feeder, MultiDataset
import torch
from torch import nn
from net.MultiView import MCL_model_v2 as MCL_model
# from net.MCL_backbone.mcl_stgcn import MCL_STGCN as MCL_model
from collections import OrderedDict

# 1.读取NTU60数据集
action_names = ['drink water', 'eat meal/snack', 'brushing teeth', 'brushing hair', 'drop', 'pickup',
            'throw', 'sitting down', 'standing up', 'clapping', 'reading', 'writing',
            'tear up paper', 'wear jacket', 'take off jacket', 'wear a shoe', 'take off a shoe',
            'wear on glasses','take off glasses', 'put on a hat/cap', 'take off a hat/cap', 'cheer up',
            'hand waving', 'kicking something', 'put/take out sth', 'hopping', 'jump up',
            'make a phone call', 'playing with a phone', 'typing on a keyboard',
            'pointing to sth with finger', 'taking a selfie', 'check time (from watch)',
            'rub two hands together', 'nod head/bow', 'shake head', 'wipe face', 'salute',
            'put the palms together', 'cross hands in front', 'sneeze/cough', 'staggering', 'falling',
            'touch head', 'touch chest', 'touch back', 'touch neck', 'nausea or vomiting condition',
            'use a fan', 'punching', 'kicking other person', 'pushing other person',
            'pat on back of other person', 'point finger at the other person', 'hugging other person',
            'giving sth to other person', 'touch other person pocket', 'handshaking',
            'walking towards each other', 'walking apart from each other']
max_frame = 300
max_body = 2
max_joint = 25

folderpath = "npy_data/xsub/val"
view1_data_path = os.path.join(folderpath,"val_view1_data.npy")
view1_label_path = os.path.join(folderpath, "val_view1_label.pkl")
view2_data_path = os.path.join(folderpath,"val_view2_data.npy")
view2_label_path = os.path.join(folderpath, "val_view2_label.pkl")
view3_data_path = os.path.join(folderpath,"val_view3_data.npy")
view3_label_path = os.path.join(folderpath, "val_view3_label.pkl")

data_loader = dict()
test_view1 = Feeder(data_path=view1_data_path, label_path=view1_label_path, debug=False)
test_view2 = Feeder(data_path=view2_data_path, label_path=view2_label_path, debug=False)
test_view3 = Feeder(data_path=view3_data_path, label_path=view3_label_path, debug=False)
multi_test = MultiDataset(test_view1, test_view2, test_view3)
data_loader["test"] = torch.utils.data.DataLoader(
    dataset=multi_test,
    batch_size=16,
    # shuffle=True,
    num_workers=0
)

# 2.加载MGL模型
state_dict_path = "ntu60_MCL_model_transfer.pth"
model = MCL_model.MCL(in_channels=3,num_class=60,device_id=0,edge_importance_weighting=True,graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
weights = torch.load(state_dict_path)
weights = OrderedDict([[k.split('module.')[-1],v.cpu()] for k, v in weights.items()])
model.load_state_dict(weights)

# 3.进行推理，得到预测结果
# device_list = [0,1,2,3,4,5,6,7]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if len(device_list) > 1:
#     model = nn.DataParallel(model, device_ids=device_list)
model.to(device)
model.to(device)
result_frag = []
label_frag = []
for batch_index, batch_data in enumerate(data_loader["test"]):
    view1, view2, view3 = batch_data[0], batch_data[1], batch_data[2]
    view1_data = view1[0].float().to(device)
    view1_label = view1[1].float().to(device)
    view2_data = view2[0].float().to(device)
    view2_label = view2[1].float().to(device)
    view3_data = view3[0].float().to(device)
    view3_label = view3[1].float().to(device)
    with torch.no_grad():
        _, output = model(view1_data, view2_data, view3_data)
    # 记录test过程中的结果
    result_frag.append(output.data.cpu().numpy())
    label_frag.append(view1_label.data.cpu().numpy())

result = np.concatenate(result_frag)
softmax = torch.nn.Softmax(dim=1)
result_softmax = softmax(torch.from_numpy(result)).numpy()
true_label = np.concatenate(label_frag)
print("result:")
print(result_softmax)
print("true_label:")
print(true_label)
rank = result_softmax.argsort()
print(rank)
predict_label = np.array([rank[i,-1] for i, l in enumerate(true_label)])
print("准确率Top1：",sum(predict_label==true_label)/predict_label.shape[0])

# 构建字典存储每个样本的top5概率大小
result_dict = {}
for i, l in enumerate(true_label):
    # samples = []
    sample = {}
    for index in range(1,6,1):
        sample['Top'+str(index)+":"+action_names[rank[i,-index]]] = result_softmax[i, rank[i,-index]]*100
        # samples.append(sample)
    result_dict['sample-'+str(i)+'-'+action_names[i%60]] = sample
print(result_dict)
np.save('top5_occlusion.npy', result_dict)




# 4.绘制混淆矩阵（confusion matrix）
cm = confusion_matrix(true_label, predict_label)
np.savetxt("result/cm.csv",cm,delimiter=',')
print(cm.shape)
cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
fig = plt.figure(figsize=(14, 12))
plt.imshow(cm, interpolation='nearest')  # 更换配色：cmap=plt.cm.Set3
# plt.title("MGL Confusion Matrix")
plt.colorbar()   # 热力图渐变色条
num_action = np.array(range(len(action_names)))
plt.xticks(num_action, action_names, rotation=90,fontproperties = 'Times New Roman',fontsize=7)   # 将标签印在x轴上
plt.yticks(num_action, action_names,fontproperties = 'Times New Roman',fontsize=7)   # 将标签印在y轴上
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.savefig('result/cm.png', dpi=1200)




