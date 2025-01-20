import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import csv
from mydata import CT
from test_loss import get_lossYonly, get_lossYsingle, get_MRR, get_f1_recall_pre, getloss, get_ea
from RCMLS import (Origin_RCML, Personal_RCML, Statistics_RCML, Normal_RCML, Dir_RCML, BaseMLP, BaseMLP_Share,
                   Originboth_RCML, TMC)


##  -------医疗数据--------------------
def train1(model_lst):
    models = dict()
    for i, model in enumerate(model_lst):
        models[i] = model(num_views, dims, num_classes)  # 模型实例化
        models[i] = models[i].to(device)
    epochs = 260
    for epoch in range(1, epochs + 1):
        #print(torch.cuda.memory_summary(device=None, abbreviated=False))
        print(f'====> {epoch}')
        for X, Y, indexes in train_loader:
            for v in range(num_views):
                X[v] = X[v].to(device)
                Y[v] = Y[v].to(device)  # !!
            Y['syn'] = Y['syn'].to(device)

            for model in models.values():
                if model.name == 'BaseMLP_Share':
                    train_mlpshare(X, Y, model)
                else:
                    train2(X, Y, model, epoch, device)
                # 每次训练后保存模型的参数
                if model.name == 'Dir_RCML':
                    current_params = {name: param.clone().detach() for name, param in model.named_parameters()}
                    params_list.append(current_params)
    return models

# 除BaseMLP_Share的train
def train2(X, Y, model, epoch, device):
    if model.name == 'Origin_RCML':
        loss_f = get_lossYonly
    elif model.name == 'BaseMLP':
        loss_f = getloss
    else:
        loss_f = get_lossYsingle
    optimizer = optim.Adam(model.parameters(), lr=0.006, weight_decay=1e-5)
    gamma = 1
    #model.to(device)
    model.train()

    '''for v in range(num_views):
        X[v] = X[v].to(device)
        Y[v] = Y[v].to(device)  # !!
    Y['syn'] = Y['syn'].to(device)  # !!
'''
    if model.name == 'TMC':
        evidences, evidence_a, loss = model(X, Y['syn'], global_step=1)
    else:
        evidences, evidence_a = model(X)

    '''if model.name == 'BaseMLP':
        evidence_a = get_ea(evidences, model.num_classes)'''

    # 初始化各模型损失函数
    if model.name == 'Origin_RCML':
        loss = loss_f(evidences, evidence_a, Y['syn'], epoch, num_classes,
                      annealing_step=50, gamma=gamma, device=device)
    elif model.name == 'BaseMLP':
        loss = loss_f(evidences, num_classes, Y)
    elif model.name == 'TMC':
        loss = loss
    else:
        loss = loss_f(evidences, evidence_a, Y, epoch, num_classes,
                      annealing_step=50, gamma=gamma, device=device)
    optimizer.zero_grad()
    loss.backward()
    max_norm = 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()


def train_mlpshare(X, Y, model):
    optimizer = optim.Adam(model.parameters(), lr=0.006, weight_decay=1e-5)
    #model.to(device)
    model.train()
    loss = nn.CrossEntropyLoss()
    # 各模态Y来监督
    for v in range(num_views):
        #X[v] = X[v].to(device)
        #Y[v] = Y[v].to(device)  # !!
        evidence = model(X[v])
        loss_value = loss(evidence, Y[v])
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
    # 最终的Y来监督
    evidence = model(X[0])
    loss_value = loss(evidence, Y['syn'])
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()


def eval_model(mrr, recall, X, Y, a, model, device, times):
    '''if model.name == 'Dir_RCML':
        PATH = '/Users/liangchenmeijin/Desktop/计算机相关/RCML-main/1/data/{}_{}.pkl'.format(model.name,times)
        torch.save(model.state_dict(), PATH)'''
    #if (times == 1) and (model.name == 'Dir_RCML'):
        #PATH = '/Users/liangchenmeijin/Desktop/计算机相关/RCML-main/1/data1/true_{}_{}.pkl'.format(model.name, times)
        #torch.save(model.state_dict(), PATH)
    model.eval()
    # Recall, MRRs, num = 0, 0, 0

    ##num_correct, num_sample = 0, 0
    for v in range(num_views):
        X[v] = X[v].to(device)
        Y[v] = Y[v].to(device)
    Y['syn'] = Y['syn'].to(device)  # !!'''
    with torch.no_grad():
        if model.name == 'BaseMLP_Share':
            evidence_a = model(X[0])
        elif model.name == 'TMC':
            evidences, evidence_a, _ = model(X, Y['syn'], global_step=1)
        else:
            evidences, evidence_a = model(X)
        if model.name == 'BaseMLP':
            evidence_a = get_ea(evidences, model.num_classes)  # 通过投票得到evidence_a

        MRR = get_MRR(evidence_a, Y['syn'])
        Recall = get_f1_recall_pre(evidence_a, Y['syn'])
        mrr[model.name] += MRR
        recall[model.name] += Recall
        for i in range(num_views):
            if model.name != 'Origin_RCML':
                # 每个模态的预测
                if model.name == 'BaseMLP_Share':
                    _, Y_pre = torch.max(evidence_a, dim=1)
                else:
                    _, Y_pre = torch.max(evidences[i], dim=1)
                a[model.name][0][i] += (Y_pre == Y[i]).sum().item()
                a[model.name][1][i] += Y[i].shape[0]
        _, Y_pre = torch.max(evidence_a, dim=1)
        a[model.name][0]['syn'] += (Y_pre == Y['syn']).sum().item()
        a[model.name][1]['syn'] += Y['syn'].shape[0]


# print('====> total_acc: {:.4f}'.format(num_correct['syn'] / num_sample['syn']))
def get_mean_final(MRRs):
    # print(MRRs)
    mean_MRR, final_MRR = torch.split(MRRs, [6, 1], dim=1)
    # print('mean_MRR,final_MRR:',mean_MRR,final_MRR)
    mean_MRR = torch.mean(mean_MRR, dim=1)
    final_MRR = final_MRR.squeeze()

    return mean_MRR, final_MRR


def main_eval(a, models, MRRs, Recall, acc, times=0):
    for i, model in enumerate(models.values()):
        '''if model.name == 'Dir_RCML':
            PATH = '/Users/liangchenmeijin/Desktop/计算机相关/RCML-main/1/data/true_{}_{}.pkl'.format(model.name,times)
            torch.save(model.state_dict(), PATH)'''
        num = 0
        print(model.name)
        model.eval()
        # mrr[model.name] =0
        # recall[model.name] =0
        c, d = dict(), dict()
        for v in range(6):
            c[v], d[v] = 0, 0
        c['syn'], d['syn'] = 0, 0
        a[model.name] = [c, d]

        for X, Y, indexes in test_loader:
            num += 1
            for v in range(num_views):
                X[v] = X[v].to(device)
                Y[v] = Y[v].to(device)
            Y['syn'] = Y['syn'].to(device)  # !!'''
            with torch.no_grad():
                if model.name == 'BaseMLP_Share':
                    evidence_a = model(X[0])
                elif model.name == 'TMC':
                    evidences, evidence_a, _ = model(X, Y['syn'], global_step=1)
                else:
                    evidences, evidence_a = model(X)

                if model.name == 'BaseMLP':
                    evidence_a = get_ea(evidences, model.num_classes)

                if model.name != 'BaseMLP_Share':
                    for v in range(num_views):
                        # print('model:',model.name)
                        MRRs[i][v] = MRRs[i][v] + get_MRR(evidences[v], Y[v])
                        Recall[i][v] = Recall[i][v] + get_f1_recall_pre(evidences[v], Y[v])
                MRRs[i][-1] = MRRs[i][-1] + get_MRR(evidence_a, Y['syn'])
                Recall[i][-1] = Recall[i][-1] + get_f1_recall_pre(evidence_a, Y['syn'])

                for j in range(num_views):
                    if model.name != 'BaseMLP_Share':
                        _, Y_pre = torch.max(evidences[j], dim=1)
                        a[model.name][0][j] += (Y_pre == Y[j]).sum().item()
                        a[model.name][1][j] += Y[j].shape[0]

                _, Y_pre = torch.max(evidence_a, dim=1)
                a[model.name][0]['syn'] += (Y_pre == Y['syn']).sum().item()
                a[model.name][1]['syn'] += Y['syn'].shape[0]

        MRRs[i] = MRRs[i] / num
        Recall[i] = Recall[i] / num
        sum_acc = 0
        if model.name == 'BaseMLP_Share':
            # 计算模态平均预测准确性
            print('====> mean_self_acc: 0')
        else:
            for v in range(num_views):
                sum_acc += (a[model.name][0][v] / a[model.name][1][v])
                # print('====> own_{}_acc: {:.4f}'.format(own, num_correct[v] / num_sample[v]))
            print('====> mean_self_acc: {:.4f}'.format(sum_acc / num_views))
            acc[i][0] = sum_acc / num_views  # 平均模态预测
        print('====> total_acc: {:.4f}'.format(a[model.name][0]['syn'] / a[model.name][1]['syn']))
        acc[i][1] = a[model.name][0]['syn'] / a[model.name][1]['syn']  # 融合预测

    # print(MRRs,Recall)
    mean_MRR, final_MRR = get_mean_final(MRRs)
    mean_Recall, final_Recall = get_mean_final(Recall)

    return acc, mean_MRR, final_MRR, mean_Recall, final_Recall


dataset = CT()
# dataset = HandWritten()
num_samples = len(dataset)
num_classes = dataset.num_classes
num_views = dataset.num_views
dims = dataset.dims
index = np.arange(num_samples)
np.random.shuffle(index)
train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]

# create a test set with conflict instances
dataset.postprocessing(test_index, addNoise=False, sigma=0.5, ratio_noise=0.1, addConflict=False,
                       ratio_conflicts=[0.3, 0.3, 0.3])

dataset.postprocessing(train_index, addNoise=False, sigma=0.1, ratio_noise=0.1, addConflict=False,
                       ratio_conflicts=[0.1, 0.7, 0.8])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(Subset(dataset, train_index), batch_size=200, shuffle=True)
test_loader = DataLoader(Subset(dataset, test_index), batch_size=200, shuffle=False)

'''model_lst=[Origin_RCML, Originboth_RCML, Personal_RCML, Statistics_RCML,Normal_RCML,Dir_RCML,BaseMLP,
           BaseMLP_Share, TMC]'''
model_lst = [Origin_RCML, Originboth_RCML, Personal_RCML, Statistics_RCML,Normal_RCML,Dir_RCML,BaseMLP,
           BaseMLP_Share, TMC]


def load_params(file_path):
    return torch.load(file_path)


'''def average_params(param_list):
    # 假设所有模型参数的结构相同
    average_params = {}
    for key in param_list[0]:
        stacked_params = torch.stack([params[key] for params in param_list])
        average_params[key] = torch.mean(stacked_params, dim=0)
    return average_params'''


def main(times=0):
    models = train1(model_lst)

    '''num_correct = dict()
    num_sample = dict()
    for v in range(6):
        num_correct[v] = 0
        num_sample[v] = 0
    num_correct['syn'] = 0
    num_sample['syn'] = 0'''
    num = 0
    a, mrr, recall = dict(), dict(), dict()
    Recalls, MRRs = torch.zeros(9, 7), torch.zeros(9, 7)
    acc = torch.zeros(9, 2)
    acc, mean_MRR, final_MRR, mean_Recall, final_Recall = main_eval(a, models, MRRs, Recalls, acc, times)

    for i, model in enumerate(models.values()):
        with open('result55.csv', mode='a', newline='') as file:
            # 创建writer对象
            writer = csv.writer(file)
            if model.name == 'Statistics_RCML':
                writer.writerow(
                    [models[i].name, acc[i][0].item(), mean_MRR[i].item(), mean_Recall[i].item(), acc[i][1].item(),
                     final_MRR[i].item(), final_Recall[i].item(), model.e_parameters.detach().tolist()])
            else:
                writer.writerow(
                    [models[i].name, acc[i][0].item(), mean_MRR[i].item(), mean_Recall[i].item(), acc[i][1].item(),
                     final_MRR[i].item(), final_Recall[i].item()])
    print('finish {} time!'.format(times))

    '''for model in models.values():
        mrr[model.name] =0
        recall[model.name] =0
        c,d=dict(),dict()
        for v in range(6):
            c[v],d[v] = 0,0
        c['syn'],d['syn'] = 0,0
        a[model.name]=[c,d]



    for X, Y, indexes in test_loader:
        num+=1
        for model in models.values():
            eval_model(mrr,recall,X,Y,a,model,device,times)'''

    '''for model in models.values():
        with open('true_data.csv', mode='a', newline='') as file:
            # 创建writer对象
            writer = csv.writer(file)
            sum_acc = 0
            if model.name != 'Origin_RCML':
                for v in range(num_views):
                    sum_acc += (a[model.name][0][v] / a[model.name][1][v])
                    #print('====> own_{}_acc: {:.4f}'.format(own, num_correct[v] / num_sample[v]))
                print('====> mean_self_acc: {:.4f}'.format(sum_acc / num_views))
                if model.name == 'Statistics_RCML':
                    writer.writerow([model.name, sum_acc / num_views, a[model.name][0]['syn'] / a[model.name][1]['syn'],
                                     (mrr[model.name] / num).item(), recall[model.name] / num, model.e_parameters])
                else:
                    writer.writerow([model.name, sum_acc / num_views, a[model.name][0]['syn'] / a[model.name][1]['syn'],
                                     (mrr[model.name] / num).item(), recall[model.name] / num])
            else:
                writer.writerow([model.name, '/',a[model.name][0]['syn'] / a[model.name][1]['syn'],
                                (mrr[model.name] / num).item(), recall[model.name] / num])
            print('====> total_acc: {:.4f}'.format(a[model.name][0]['syn']/ a[model.name][1]['syn']))
            print('mena_MRR:', mrr[model.name] / num)
            print('mena_Recall:', recall[model.name] / num)'''


## 'statistic', 'personal'
# main(atten='statistic')
np.set_printoptions(precision=4, suppress=True)
torch.autograd.set_detect_anomaly(True)
#file_paths=[]
for i in range(15):
    #file_paths.append('/Users/liangchenmeijin/Desktop/计算机相关/RCML-main/1/data/true_Dir_RCML_{}.pkl'.format(i))
    params_list = []
    main(times=i)
'''
# 计算参数的平均值   这是正确的！！！
avg_params = {}

for name in params_list[0].keys():
    # 堆叠每次运行的同一参数
    stacked_params = torch.stack([params[name] for params in params_list])
    # 计算平均值
    avg_params[name] = torch.mean(stacked_params, dim=0)

model=Dir_RCML(num_views, dims, num_classes)
#PATH = '/Users/liangchenmeijin/Desktop/计算机相关/RCML-main/1/data/Dir_RCML_true_average.pkl'


with torch.no_grad():
    for name, param in model.named_parameters():
        param.copy_(avg_params[name])


PATH = '/Users/liangchenmeijin/Desktop/计算机相关/RCML-main/1/data/true_{}_avg1.pkl'.format(model.name)
torch.save(model.state_dict(), PATH)'''
#载入保存的模型参数
#model.load_state_dict(torch.load(PATH))


