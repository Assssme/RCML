import torch
from sklearn.metrics import recall_score
import torch.nn as nn
import torch.nn.functional as F

def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None, useKL=True):
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    if not useKL:
        return loglikelihood

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device, useKL=True):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    # A是Lace
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    if not useKL:
        return A

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    # 退火系数 * KL散度
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    return torch.mean(loss)


def edl_log_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = edl_loss(torch.log, target, alpha, epoch_num, num_classes, annealing_step, device)
    return torch.mean(loss)


def edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device)
    return torch.mean(loss)


def get_dc_loss(evidences, device):  # 计算冲突程度
    num_views = len(evidences)-1
    batch_size, num_classes = evidences[0].shape[0], evidences[0].shape[1]
    p = torch.zeros((num_views, batch_size, num_classes)).to(device)
    u = torch.zeros((num_views, batch_size)).to(device)
    for v in range(num_views):
        alpha = evidences[v] + 1   # 迪利克雷参数
        S = torch.sum(alpha, dim=1, keepdim=True)
        p[v] = alpha / S  # evidence + 1 /S
        u[v] = torch.squeeze(num_classes / S)
    dc_sum = 0
    lst = []
    for i in range(num_views):
        # projected distance between different views
        pd = torch.sum(torch.abs(p - p[i]) / 2, dim=2) / (num_views - 1)  # (num_views, batch_size)
        # conjunctive certainty between different views
        cc = (1 - u[i]) * (1 - u)  # (num_views, batch_size)
        dc = pd * cc  # degree of conflict 单视图总的冲突程度
        lst.append(dc)
        dc_sum = dc_sum + torch.sum(dc, dim=0)
    '''ct = torch.stack(lst, dim=0)
    plt.imshow(torch.mean(ct, dim=2))
    plt.colorbar()
    plt.tight_layout()
    plt.show()'''
    dc_sum = torch.mean(dc_sum)  # 视图平均冲突程度
    #print('视图平均冲突程度:',dc_sum)
    return dc_sum


def get_syn_loss(evidence_a, target, epoch_num, num_classes, annealing_step, gamma, device):  # !!
    target = F.one_hot(target, num_classes)
    alpha_a = evidence_a + 1
    loss_acc = edl_digamma_loss(alpha_a, target, epoch_num, num_classes, annealing_step, device)
    return loss_acc

# 自身Y和Y['syn']一起监督
def get_lossYsingle(evidences, evidence_a, targets, epoch_num, num_classes, annealing_step, gamma, device, para=None):  # !!
    #para = nn.Parameter(nn.functional.softmax(para))
    #weight = para.dim
    loss_acc = get_syn_loss(evidence_a, targets['syn'], epoch_num, num_classes, annealing_step, gamma,
                            device)  # !!gamma
    # 第一次加的总的targets['syn']
    #weight = nn.Softmax(para).dim
    #for v in range(1, len(evidences)):   # len(evidences)为模态数
    for v in range(1, len(evidences)-1):   #>>>>
        target_v = F.one_hot(targets[v], num_classes)
        # !!  每个view不仅evidence不一样，判断的类别target也不一样
        # 第二次加的targets[v]
        alpha = evidences[v] + 1
        #loss_acc += edl_digamma_loss(alpha, target_v, epoch_num, num_classes, annealing_step, device)  # !!
        loss_acc += edl_digamma_loss(alpha, target_v, epoch_num, num_classes, annealing_step, device)  # !!
    ##loss_acc = loss_acc / (len(evidences) + 1)
   ## loss_acc = loss_acc / (len(evidences))  #>>>>
    #loss = loss_acc + gamma * get_dc_loss(evidences, device)
    loss = loss_acc / (len(evidences))

    return loss

# 只有Y['syn']监督+最小冲突(RCML的loss)
def get_lossYonly(evidences, evidence_a, target, epoch_num, num_classes, annealing_step, gamma, device):
    target = F.one_hot(target, num_classes)
    alpha_a = evidence_a + 1
    loss_acc = edl_digamma_loss(alpha_a, target, epoch_num, num_classes, annealing_step, device)
    for v in range(len(evidences)):
        alpha = evidences[v] + 1
        loss_acc += edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device)
    #loss = loss_acc / (len(evidences))
    loss = loss_acc + gamma * get_dc_loss(evidences, device)
    return loss

def get_ea(e,classes):
    _,ypre = torch.max(e[0],dim=1)
    tol = ypre.unsqueeze(1)

    for i in range(1,6):
        _,ypre = torch.max(e[i],dim=1)
        tol = torch.cat((tol, ypre.unsqueeze(1)), dim=1)
    tol_vote = tol.mode(dim=1).values.squeeze()
    return F.one_hot(tol_vote, classes).float().requires_grad_()

def getloss(evidences, classes, Y):
    # 修正ea
    evidence_a = get_ea(evidences,classes)
    loss = nn.CrossEntropyLoss()
    loss_acc = loss(evidence_a, Y['syn'])
    for i in range(len(Y)-1):
        loss_acc += loss(evidences[i], Y[i])
    loss_acc = loss_acc #/ (len(evidences))
   # loss_acc = loss(evidence_a, Y['syn'])

    return loss_acc


def get_MRR(evidence,Y_true):  #是top1的
    MRR = 0
    _,idx = torch.sort(evidence, descending=True)
    for i in range(len(Y_true)):
        a = torch.where(idx[i]==Y_true[i])[0]   # 由高到低排序后，Yture所在的索引
        mrr = 1/(a+1)
        #print(mrr)
        MRR+=mrr
    MRR = MRR/len(Y_true)
    return MRR

def get_f1_recall_pre(evidence,Y_true): #是所有的
    _,Y_pre = torch.max(evidence, dim=1)
    average = 'micro'

    recall = recall_score(Y_true.cpu(), Y_pre.cpu(), average=average) # 这个就是top1的

    return recall

#ce_loss(y, alpha_a, self.classes, global_step, self.lambda_epochs)

def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p.long(), num_classes=torch.tensor(c, dtype=torch.int64))
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return (A + B)


