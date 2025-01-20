
from torch.utils.data import DataLoader, Subset
from mydata import CT,CT1
from test_loss import get_dc_loss

from RCMLS import Dir_RCML

import torch
import matplotlib as mpl
import argparse
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def get_dc_loss(evidences, e_a, device):
    num_views = len(evidences)-1
    batch_size, num_classes = evidences[0].shape[0], evidences[0].shape[1]
    p = torch.zeros((num_views, batch_size, num_classes)).to(device)
    u = torch.zeros((num_views, batch_size)).to(device)
    for v in range(num_views):
        alpha = evidences[v] + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        p[v] = alpha / S
        u[v] = torch.squeeze(num_classes / S)
    alpha = e_a + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    u_a = torch.squeeze(num_classes / S)

    dc_sum = 0
    lst = []
    for i in range(num_views):
        pd = torch.sum(torch.abs(p - p[i]) / 2, dim=2) / (num_views - 1)  # (num_views, batch_size)
        cc = (1 - u[i]) * (1 - u)  # (num_views, batch_size)
        dc = pd * cc
        lst.append(dc)
        dc_sum = dc_sum + torch.sum(dc, dim=0)

    dc_sum = torch.mean(dc_sum)
    return u,u_a

parse = argparse.ArgumentParser()
parse.add_argument('--patient_info',type=str,default = "{\"is_ct\":1,\"pathological_type\":6,\"er_value\":0.8,\"pr_value\":0,\"cerbb_2\":1,\"her2_fish\":2,\"histological_grade\":5,\"ki67_value\":30,\"rs_21_gene\":25,\"age\":64,\"tumor_phase\":9,\"node_phase\":0}")
parse.add_argument('--model_path',type=str,default = 'true_Dir_RCML_avg.pkl')

args, unknown = parse.parse_known_args()

patient_info = json.loads(args.patient_info)
new_data = pd.DataFrame([patient_info])

all_data=pd.read_csv('/Users/liangchenmeijin/Desktop/计算机相关/RCML-main/1/data/R2V_MainUsers1.csv', encoding="utf-8", index_col=12).copy()
all_data = all_data.drop('Unnamed: 0', axis=1).copy()

##############
idx = 49         #  178([4, 4, 4, 85, 4, 85, 4])  62, 145,-10
##############
# 找一个3不是P最高的，且3T不是finalT  ---> 62
# 再找一个分布和mean分布类似的，1T最高且是finalT   ---> 50 49 33
torch.manual_seed(idx)
#new_data = pd.DataFrame(all_data.iloc[idx]).T#.iloc[1:]#[:13]
new_data = all_data.iloc[[idx],:13]#[:13]
# 4308个train样本，1077个test样本 +1
dataset = CT1(new_data)
num_samples = len(dataset)
num_classes = dataset.num_classes  #方案个数
num_views = dataset.num_views #模态数
dims = dataset.dims
index = np.arange(num_samples)
np.random.shuffle(index)
model = Dir_RCML(num_views, dims, num_classes)

# 载入保存的模型参数
PATH = '/Users/liangchenmeijin/Desktop/计算机相关/RCML-main/1/data/true_Dir_RCML_avg.pkl'
model.load_state_dict(torch.load(PATH))

#train_index, _ = index[:int(0.8 * num_samples-1)], index[int(0.8 * num_samples-1):]

#torch.manual_seed(idx)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#train_loader = DataLoader(Subset(dataset, train_index), batch_size=200, shuffle=True)
test_index = [-1]
test_loader = DataLoader(Subset(dataset, test_index), batch_size=200, shuffle=False)
#print(dataset[test_index])
# us,ua,priority,self_acc,total_acc

models = dict()
models[0] = model

a, mrr, recall = dict(), dict(), dict()
Recall, MRRs = torch.zeros(9, 7), torch.zeros(9, 7)
acc = torch.zeros(9, 2)
for i, model in enumerate(models.values()):
    num = 0
    print(model.name)
    model.eval()
    c, d = dict(), dict()
    for v in range(6):
        c[v], d[v] = 0, 0
    c['syn'], d['syn'] = 0, 0
    a[model.name] = [c, d]

    for X, Y, indexes in test_loader:
        #print('X', X)
        #print('Y', Y)
        #print(indexes)
        num += 1
        for v in range(num_views):
            X[v] = X[v].to(device)
            Y[v] = Y[v].to(device)
        Y['syn'] = Y['syn'].to(device)  # !!
        with torch.no_grad():
            result = dict()
            evidences, evidence_a, dir_fuse_weight, sts_w = model(X)
            u, u_a = get_dc_loss(evidences, evidence_a, device)
            dir_w = dir_fuse_weight[0] #[round(x, 4) for x in dir_fuse_weight[0]]
            us = u.squeeze().tolist() #[round(1 - y, 4) for y in u.squeeze().tolist()]  # 不确定性
            print('uncertanties:', us)
            #print('fianl_uncertanty:', u_a)
            print('dir_fuse_weight:', dir_w)
            result['Dir_w'] = dir_fuse_weight[0].tolist()
            prd_lst = []

            for v in range(num_views):
                _, Y_pre = torch.max(evidences[v], dim=1)
                a[model.name][0][v] += (Y_pre == Y[v]).sum().item()
                a[model.name][1][v] += Y[v].shape[0]
                #print('view and pre:', v, Y_pre)
                result['{}view_predict'.format(v)] = Y_pre.item()
                prd_lst.append(Y_pre.item())
            _, Y_pre = torch.max(evidence_a, dim=1)
            a[model.name][0]['syn'] += (Y_pre == Y['syn']).sum().item()
            a[model.name][1]['syn'] += Y['syn'].shape[0]
            result['final_predict:'] = Y_pre.item()
            prd_lst.append(Y_pre.item())
            # result['final_uncertainty'] = u_a.item()
            # print('final_predict:',Y_pre[0])
            json_result = json.dumps(result)

            #print(json_result)
            print('all_prd:', prd_lst)


#prd_lst=[11, 11, 11, 11, 11, 11, 11]
labels = ['1(T{})'.format(prd_lst[0]), '2(T{})'.format(prd_lst[1]), '3(T{})'.format(prd_lst[3]),
          '4(T{})'.format(prd_lst[3]), '5(T{})'.format(prd_lst[4]), '6(T{})'.format(prd_lst[5])]

# 不确定性
#us = [0.7952821254730225, 0.9840032458305359, 0.9388445019721985,
      #0.9239650964736938, 0.9613381028175354, 0.9728972315788269]

us = us
# 89:[0.9076306819915771, 0.9851075410842896, 0.9643663763999939, 0.9875342845916748, 0.9769834876060486, 0.9868866801261902]

# 迪利克雷权重
#var1 = torch.tensor([0.3360, 0.0672, 0.1492, 0.2629, 0.1293, 0.0554])   # priority
var1 = dir_w
# 89:[0.1016, 0.1840, 0.2878, 0.1447, 0.1525, 0.1294]

var2 = [1 - y for y in us] # 可靠性

x = np.arange(len(labels))  # x轴标签的位置
width = 0.35  # 每个柱状图的宽度
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
mpl.rcParams['font.weight'] = 'bold'
fig, ax = plt.subplots()

# 绘制柱状图
rects1 = ax.bar(x - width/2, var1, width, label='Priority', color='skyblue', edgecolor='black')
rects2 = ax.bar(x + width/2, var2, width, label='Reliability', color='salmon', edgecolor='black')


# 添加标签、标题和x轴标签
ax.set_xlabel('Doctors/ final: T{}'.format(prd_lst[-1]), fontsize=12, weight='bold')
ax.set_ylabel('Values', fontsize=12, weight='bold')
ax.set_title('Priority and Reliability Comparison', fontsize=14, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10, weight='bold')

def add_value_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',  # 显示数值到小数点后两位
                    xy=(rect.get_x() + rect.get_width() / 2, height),  # 数值显示位置
                    xytext=(0, 3),  # 数值与柱顶的偏移
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color='black')

add_value_labels(rects1)
add_value_labels(rects2)
# 添加网格线提高对比度
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 添加图例
ax.legend(loc='upper right', fontsize=10)


'''prds = prd_lst
for i in range(6):
    ax.text(i, -0.1, 'prd: {}'.format(prds[i]), fontsize=12, color='black')
    '''

plt.tight_layout()
#plt.savefig("/Users/liangchenmeijin/Desktop/计算机相关/RCML-main/IJCAI--24 Formatting Instructions/images/patient1.svg", format="svg")
plt.show()

sum_acc = 0

'''for v in range(num_views):
    sum_acc += (a[model.name][0][v] / a[model.name][1][v])
    # print('====> own_{}_acc: {:.4f}'.format(own, num_correct[v] / num_sample[v]))
#print('====> mean_self_acc: {:.4f}'.format(sum_acc / num_views))
acc[i][0] = sum_acc / num_views  # 平均模态预测
#print('====> total_acc: {:.4f}'.format(a[model.name][0]['syn'] / a[model.name][1]['syn']))
acc[i][1] = a[model.name][0]['syn'] / a[model.name][1]['syn']  # 融合预测
'''

def plot_heat(data,name,cmap='YlGnBu'):
    print(data)
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    mpl.rcParams['font.weight'] = 'bold'
    #user_labels = [f'Member {i + 1}' for i in range(data.size()[0])]
    user_labels = [f'Member {i+1}' if i % 5 == 0 else '' for i in range(data.size()[0])]
    layer_labels = [f'Doctor {i+1}' for i in range(data.size()[1])]


    # 创建热力图
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(data, xticklabels=layer_labels, yticklabels=user_labels, cmap=cmap,
                     cbar_kws={'label': 'Value'})
    cbar = ax.collections[0].colorbar  # 获取颜色条
    cbar.ax.yaxis.label.set_size(14)  # 设置颜色条标签的字号
    cbar.ax.yaxis.label.set_weight('bold')

    # 设置标题
    ax.set_title(r'{}'.format(name),fontsize=30, weight='bold')
    # 调整标签字体
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12, weight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, weight='bold')

    # 显示图像
    plt.tight_layout()  # 防止标签被裁剪


def get_rs_dws(idxs):

    for n,idx in enumerate(idxs):
        new_data = all_data.iloc[[idx], :13]  # 一行
        dataset = CT1(new_data)
        test_index = [-1]
        test_loader = DataLoader(Subset(dataset, test_index), batch_size=200, shuffle=False)

        model.eval()
        for X, Y, indexes in test_loader:
            for v in range(num_views):
                X[v] = X[v].to(device)
                Y[v] = Y[v].to(device)
            Y['syn'] = Y['syn'].to(device)  # !!
            with torch.no_grad():
                result = dict()
                evidences, evidence_a, dir_fuse_weight, sts_w = model(X)
                u, u_a = get_dc_loss(evidences, evidence_a, device)
                dir_w = dir_fuse_weight[0]  # [round(x, 4) for x in dir_fuse_weight[0]]
                us = u.squeeze()  # [round(1 - y, 4) for y in u.squeeze().tolist()]  # 不确定性
                print('uncertanties:', us)
                print('fianl_uncertanty:', u_a)
                print('dir_fuse_weight:', dir_w)
                result['Dir_w'] = dir_fuse_weight[0].tolist()
                if n == 0:
                    dir_ws = dir_w.unsqueeze(0)
                    reliability = 1-u.T
                else:
                    dir_ws = torch.cat((dir_ws,dir_w.unsqueeze(0)))
                    reliability = torch.cat((reliability,1-u.T))

                prd_lst = []

                for v in range(num_views):
                    _, Y_pre = torch.max(evidences[v], dim=1)
                    #a[model.name][0][v] += (Y_pre == Y[v]).sum().item()
                    #a[model.name][1][v] += Y[v].shape[0]
                    #print('view and pre:', v, Y_pre)
                    result['{}view_predict'.format(v)] = Y_pre.item()
                    prd_lst.append(Y_pre.item())
                _, Y_pre = torch.max(evidence_a, dim=1)
                #a[model.name][0]['syn'] += (Y_pre == Y['syn']).sum().item()
                #a[model.name][1]['syn'] += Y['syn'].shape[0]
                result['final_predict:'] = Y_pre.item()
                prd_lst.append(Y_pre.item())
                # result['final_uncertainty'] = u_a.item()
                # print('final_predict:',Y_pre[0])
                json_result = json.dumps(result)

    #plot_heat(dir_ws,'Priority of members')
    #print('Priority of members:',dir_ws)
    plot_heat(reliability, 'Realiability of members')
    print('Realiability of members',reliability)
    plt.show()

def plot_heatmap(data, title, x_labels=None, y_labels=None, cmap='viridis'):
    """
    绘制热力图的函数。
    """
    user_labels = [f'Member {i + 1}' if i % 5 == 0 else '' for i in range(data.size()[0])]
    layer_labels = [f'Doctor {i + 1}' for i in range(data.size()[1])]
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt=".2f", cmap=cmap, cbar=True,
                xticklabels=layer_labels, yticklabels=user_labels, linewidths=0.5)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Members', fontsize=12)
    plt.ylabel('Instances', fontsize=12)
    plt.tight_layout()
    plt.savefig(
        "/Users/liangchenmeijin/Desktop/计算机相关/RCML-main/IJCAI--24 Formatting Instructions/images/111.svg",
        format="svg")
    plt.show()

def get_rs_dws1(idxs):
    """
    主函数，执行数据计算和热力图绘制。
    """
    dir_ws = None
    reliability = None

    for n, idx in enumerate(idxs):
        new_data = all_data.iloc[[idx], :13]
        dataset = CT1(new_data)
        test_index = [-1]
        test_loader = DataLoader(Subset(dataset, test_index), batch_size=200, shuffle=False)

        model.eval()
        for X, Y, indexes in test_loader:
            for v in range(num_views):
                X[v] = X[v].to(device)
                Y[v] = Y[v].to(device)
            Y['syn'] = Y['syn'].to(device)  # !!

            with torch.no_grad():
                result = dict()
                evidences, evidence_a, dir_fuse_weight, sts_w = model(X)
                u, u_a = get_dc_loss(evidences, evidence_a, device)
                dir_w = dir_fuse_weight[0]
                us = u.squeeze()

                # 打印信息
                print('Uncertainties:', us)
                print('Final Uncertainty:', u_a)
                print('Dir Fuse Weight:', dir_w)

                # 结果矩阵更新
                if n == 0:
                    dir_ws = dir_w.unsqueeze(0)
                    reliability = 1 - u.T
                else:
                    dir_ws = torch.cat((dir_ws, dir_w.unsqueeze(0)))
                    reliability = torch.cat((reliability, 1 - u.T))

                # 预测结果
                prd_lst = []
                for v in range(num_views):
                    _, Y_pre = torch.max(evidences[v], dim=1)
                    result['{}view_predict'.format(v)] = Y_pre.item()
                    prd_lst.append(Y_pre.item())
                _, Y_pre = torch.max(evidence_a, dim=1)
                result['final_predict:'] = Y_pre.item()
                prd_lst.append(Y_pre.item())

                json_result = json.dumps(result)

    # 绘制热力图
    plot_heatmap(dir_ws, 'Priority of Members', cmap='coolwarm')
    print('Priority of Members:', dir_ws)

    plot_heatmap(reliability, 'Reliability of Members', cmap='YlGnBu')
    print('Reliability of Members:', reliability)
#idxs = [1159, 6407, 7838, 5139, 4138,6,46,1378,45,2,79]
#idxs = [4960, 5430, 925, 4177, 3480, 3152, 620, 7094, 5838, 3478, 7905, 5384, 5037, 3221, 2342, 5512, 6268,1159, 6407, 7838, 5139, 4138,6,46,1378,45,2,79]
idxs = [7262, 7742, 7205, 2500, 7716, 2126, 256, 90, 7147, 5785, 2151, 5245, 1562, 1713, 2401, 5318, 6256, 7904, 8602, 7695,
        4960, 925, 4177, 3480, 3152, 620, 7094, 5838, 3478, 7905, 5384, 3221, 2342, 5512, 6268, 6407, 5139]
#print(idxs)
get_rs_dws(idxs)
