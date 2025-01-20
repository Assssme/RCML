import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.dirichlet as dirichlet


# 个性化加权的权重
class MLP(nn.Module):
    def __init__(self, dims, num_classes, num_views):
        super(MLP, self).__init__()
        #self.num_views = num_views
        self.num_classes = num_classes
        self.net = nn.Sequential(nn.Flatten(),
                    nn.Linear(dims[0], 256),
                    nn.ReLU(),
                    nn.Linear(256, num_views), nn.Softplus())


    def forward(self, x):
        h = self.net(x)
        '''for i in range(1, len(self.net)):
            h = self.net[i](h)'''
        return h

class EvidenceCollector(nn.Module):
    def __init__(self, dims, num_classes,ratio=4):
        super(EvidenceCollector, self).__init__()
        self.num_layers = len(dims)
        self.net = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
            self.net.append(nn.ReLU())
            #self.net.append(nn.Dropout(0.1))

        # ！！！不加MLP的话，这里就是一个线性层，evidence主要是loss那里体现
        #self.net.append(nn.Linear(dims[self.num_layers - 1], dims[self.num_layers - 1]//ratio, bias=False))
        #self.net.append(nn.Flatten())  # !!!
        self.net.append(nn.Linear(dims[self.num_layers - 1], 256, bias=False))
        self.net.append(nn.ReLU())
        #self.net.append(nn.Dropout(0.1))
        #self.net.append(nn.Linear(dims[self.num_layers - 1]//ratio, dims[self.num_layers - 1], bias=False))
        self.net.append(nn.Linear(256,  num_classes, bias=False))
        ##self.net.append(nn.Linear(dims[self.num_layers - 1], num_classes))
        self.net.append(nn.Softplus())

    def forward(self, x):
        h = self.net[0](x)
        for i in range(1, len(self.net)): #>>!!
            h = self.net[i](h)
        return h

# 差一个，相邻加和融合evidence，单模态evidence和融合模态evidence_a共同监督
class Originboth_RCML(nn.Module):
    def __init__(self, num_views, dims, num_classes):
        super(Originboth_RCML, self).__init__()
        self.name = 'Originboth_RCML'
        #self.atten=atten
        self.num_views = num_views
        self.num_classes = num_classes
        #self.atten_para = nn.Parameter(torch.ones(num_views, dims[0][0]))
        self.EvidenceCollectors = nn.ModuleList([EvidenceCollector(dims[i], self.num_classes) for i in range(self.num_views)])

    def forward(self, X):
        # get evidence
        evidences = dict()
        for v in range(self.num_views):
            evidences[v] = self.EvidenceCollectors[v](X[v])
        # average belief fusion
        evidence_a = evidences[0]
        for i in range(1, self.num_views):
            evidence_a = (evidences[i] + evidence_a) / 2
        return evidences, evidence_a

# 相邻加和得到融合evidence_a，只用evidence_a监督
class Origin_RCML(nn.Module):
    def __init__(self, num_views, dims, num_classes):
        super(Origin_RCML, self).__init__()
        self.name = 'Origin_RCML'
        #self.atten=atten
        self.num_views = num_views
        self.num_classes = num_classes
        #self.atten_para = nn.Parameter(torch.ones(num_views, dims[0][0]))
        self.EvidenceCollectors = nn.ModuleList([EvidenceCollector(dims[i], self.num_classes) for i in range(self.num_views)])
        '''if atten == 'personal':
            self.attentions = nn.ModuleList(
                [final_attention(dims[i], num_hiddens1=6, classes=self.num_classes, kv_pairs=dims[i][0]) for i in
                 range(self.num_views)])'''

    def forward(self, X):
        # get evidence
        evidences = dict()
        for v in range(self.num_views):
            evidences[v] = self.EvidenceCollectors[v](X[v])
        # average belief fusion
        evidence_a = evidences[0]
        for i in range(1, self.num_views):
            evidence_a = (evidences[i] + evidence_a) / 2
        return evidences, evidence_a

# 个性化加权融合evidence
class Personal_RCML(nn.Module):
    # 动态加权
    def __init__(self, num_views, dims, num_classes):
        super(Personal_RCML, self).__init__()
        self.name = 'Personal_RCML'
        #self.atten = atten
        self.num_views = num_views
        self.num_classes = num_classes
        #self.e_parameters = nn.Parameter(torch.normal(mean=torch.full((num_views,),1.0/num_views)
                                                      #,std=torch.full((num_views,),0.01)))
        self.e_parameters = nn.Parameter(
             torch.tensor([0.2910, 0.1555, 0.1661, 0.1386, 0.1071, 0.1417]))
        #self.atten_para = nn.Parameter(torch.ones(num_views, dims[0][0]))
        self.EvidenceCollectors = nn.ModuleList(
            [EvidenceCollector(dims[i], self.num_classes) for i in range(self.num_views)]
        + [MLP(dims[0], num_classes, num_views)])
        '''if atten == 'personal':
            self.attentions = nn.ModuleList(
                [final_attention(dims[i], num_hiddens1=6, classes=self.num_classes, kv_pairs=dims[i][0]) for i in
                 range(self.num_views)])'''

    def forward(self, X):
        # get evidence
        evidences = dict()
        #attention = dict()
        #evidences['wg'] = self.MLP(X[0])
        evidences['wg'] = self.EvidenceCollectors[-1](X[0])

        evidences[0] = self.EvidenceCollectors[0](X[0])
        evidence_a = evidences[0].clone() * evidences['wg'][:, 0:0 + 1]
        for v in range(1, self.num_views):
            evidences[v] = self.EvidenceCollectors[v](X[v])
            evidence_a += evidences[v].clone() * evidences['wg'][:, v:v + 1]

        '''for v in range(self.num_views):
            # 这里是对X的注意力机制，但不用
            evidences[v] = self.EvidenceCollectors[v](X[v]) * evidences['wg'][:, v:v + 1]  ## 修改这里！！

        evidence_a = evidences[0]
        for i in range(1, self.num_views):
            evidence_a += evidences[i]'''
        return evidences, evidence_a

#静态
class Statistics_RCML(nn.Module):
    # 静态加权
    def __init__(self, num_views, dims, num_classes):
        super(Statistics_RCML, self).__init__()
        self.name = 'Statistics_RCML'
        #self.atten = atten
        self.num_views = num_views
        self.num_classes = num_classes
        #self.e_parameters = nn.Parameter(torch.normal(mean=torch.full((num_views,),1.0/num_views),
                                                      #std=torch.full((num_views,),0.01)))
        self.e_parameters = nn.Parameter(
            torch.tensor([0.2910, 0.1555, 0.1661, 0.1386, 0.1071, 0.1417]))
        self.atten_para = nn.Parameter(torch.ones(num_views, dims[0][0]))
        self.EvidenceCollectors = nn.ModuleList(
            [EvidenceCollector(dims[i], self.num_classes) for i in range(self.num_views)]
            + [MLP(dims[0], num_classes, num_views)])

    def forward(self, X):
        # get evidence
        evidences = dict()
        #attention = dict()
        evidences['wg'] = self.EvidenceCollectors[-1](X[0])

        evidences[0] = self.EvidenceCollectors[0](X[0])
        evidence_a = evidences[0].clone() * self.e_parameters[0]
        for v in range(1, self.num_views):
            evidences[v] = self.EvidenceCollectors[v](X[v])
            evidence_a += evidences[v].clone() * self.e_parameters[v]

        '''evidence_a = evidences[0] * self.e_parameters[0]

        for i in range(1, self.num_views):
            #evidence_a = (evidences[i] + evidence_a) / 2
            evidence_a += evidences[i] * self.e_parameters[i]'''

        return evidences, evidence_a

#正态分布
class Normal_RCML(nn.Module):
    def __init__(self, num_views, dims, num_classes):
        super(Normal_RCML, self).__init__()
        self.name = 'Normal_RCML'
        self.num_views = num_views
        #self.atten = atten
        self.num_classes = num_classes
        #self.e_parameters = nn.Parameter(torch.normal(mean=torch.full((num_views,),1.0/num_views),std=torch.full((num_views,),0.01)))
        #self.e_parameters = nn.Parameter(nn.functional.softmax(torch.tensor([82.824859,76.384181,77.062147,75.197740,72.542373,75.423729])))
        self.e_parameters = nn.Parameter(
            torch.tensor([0.2910, 0.1555, 0.1661, 0.1386, 0.1071, 0.1417]))
        #self.atten_para = nn.Parameter(torch.ones(num_views, dims[0][0]))
        self.EvidenceCollectors = nn.ModuleList(
            [EvidenceCollector(dims[i], self.num_classes) for i in range(self.num_views)]
        + [MLP(dims[0], num_classes, num_views)])
        '''if atten == 'personal':
            self.attentions = nn.ModuleList([final_attention(dims[i], num_hiddens1=6, classes=self.num_classes,
                                                             kv_pairs=dims[i][0]) for i in range(self.num_views)])
'''
    def forward(self, X):
        # get evidence
        evidences = dict()
        #attention = dict()
        evidences['wg'] = self.EvidenceCollectors[-1](X[0])
        fuse_weight = torch.FloatTensor(evidences['wg'].size()).normal_().cuda()
        #std = evidences['wg'].mul(0.5).exp_()#.cuda()
        std = torch.minimum(evidences['wg'].mul(0.5), torch.tensor(1.0))
        #std = torch.mean(evidences['wg'], dim=0)
        #std = torch.sqrt(nn.functional.softmax(evidences['wg'])+0.5)
        #fuse_weight = nn.functional.softplus(fuse_weight.mul(self.e_parameters).add_(std))
        fuse_weight = nn.functional.softplus(fuse_weight.mul(std).add_(self.e_parameters))
        # fuse_weight = self.e_parameters+std
        evidences[0] = self.EvidenceCollectors[0](X[0])
        evidence_a = evidences[0].clone() * fuse_weight[:, 0:0 + 1]
        for v in range(1, self.num_views):
            evidences[v] = self.EvidenceCollectors[v](X[v])  # * dir_fuse_weight[:, v:v + 1]  ## 修改这里！！
            evidence_a += evidences[v].clone() * fuse_weight[:, v:v + 1]

        '''for v in range(self.num_views):
            evidences[v] = self.EvidenceCollectors[v](X[v]) * fuse_weight[:, v:v + 1]  ## 修改这里！！

        evidence_a = evidences[0]
        for i in range(1, self.num_views):
            evidence_a += evidences[i]'''
        return evidences, evidence_a
#迪利克雷分布
class Dir_RCML(nn.Module):
    def __init__(self, num_views, dims, num_classes):
        super(Dir_RCML, self).__init__()
        self.name = 'Dir_RCML'
        self.num_views = num_views
        #self.atten = atten
        self.num_classes = num_classes
        #self.e_parameters = nn.Parameter(torch.normal(mean=torch.full((num_views,),1.0/num_views),std=torch.full((num_views,),0.01)))
        self.e_parameters = nn.Parameter(
            torch.tensor([0.2910, 0.1555, 0.1661, 0.1386, 0.1071, 0.1417]))
        #self.atten_para = nn.Parameter(torch.ones(num_views, dims[0][0]))
        self.EvidenceCollectors = nn.ModuleList(
            [EvidenceCollector(dims[i], self.num_classes) for i in range(self.num_views)]
        + [MLP(dims[0], num_classes, num_views)])
        '''if atten == 'personal':
            self.attentions = nn.ModuleList(
                [final_attention(dims[i], num_hiddens1=6, classes=self.num_classes, kv_pairs=dims[i][0]) for i in
                 range(self.num_views)])'''

    def forward(self, X):
        # get evidence
        evidences = dict()
        evidences['wg'] = self.EvidenceCollectors[-1](X[0])
        fuse_weight = torch.FloatTensor(evidences['wg'].size()).normal_().cuda()
        std = evidences['wg'].mul(0.5).exp_().cuda()  # sigma方差
        #std = torch.minimum(evidences['wg'].mul(0.5), torch.tensor(1.0))
        #std = torch.std(evidences['wg'],dim=0)
        #fuse_weight = nn.functional.softplus(fuse_weight.mul(self.e_parameters).add_(std))
        #fuse_weight = fuse_weight.mul(self.e_parameters).add_(std)

        fuse_weight = nn.functional.softplus(fuse_weight.mul(std).add_(self.e_parameters))
        #fuse_weight = self.e_parameters + std   # 浓度系数
        while torch.any(fuse_weight <= 0):
            fuse_weight = fuse_weight.mul(std).add_(self.e_parameters)
            #raise ValueError("concentration 参数包含非法值，请确保所有值都大于 0。")
        else:
            pass
        #views_count = torch.tensor([4533, 4112, 4355, 4037, 4313, 4123], dtype=torch.float32)

        poster = nn.functional.softplus(torch.tensor([1, 0.15120968, 0.64112903, 0, 0.55645161, 0.1733871], dtype=torch.float32))
        #Dirichlet = dirichlet.Dirichlet(fuse_weight)
        Dirichlet = dirichlet.Dirichlet((fuse_weight+poster.cuda()))
        dir_fuse_weight = Dirichlet.sample()


        #print('dir_fuse_weight:',dir_fuse_weight)
        #s_w = torch.zeros(len(evidences['wg']), 6)
        evidences[0] = self.EvidenceCollectors[0](X[0])
        evidence_a = evidences[0].clone() * dir_fuse_weight[:, 0:0 + 1]
        for v in range(1,self.num_views):
            evidences[v] = self.EvidenceCollectors[v](X[v]) #* dir_fuse_weight[:, v:v + 1]  ## 修改这里！！
            evidence_a += evidences[v].clone() * dir_fuse_weight[:, v:v + 1]


            #s_w[:,v] = evidences[v].sum(dim=1)

        '''fuse_weight = torch.FloatTensor(evidences['wg'].size()).normal_()  # .cuda()
        std = evidences['wg'].mul(0.5).exp_()  # .cuda()  # sigma方差
        fuse_weight = nn.functional.softplus(fuse_weight.mul(std).add_(self.e_parameters))
        #fuse_weight = nn.functional.softplus(fuse_weight.mul(std).add_(s_w))
        Dirichlet = dirichlet.Dirichlet((fuse_weight))
        dir_fuse_weight = Dirichlet.sample()'''


        '''evidence_a = evidences[0].clone()
        for i in range(1, self.num_views):
            #evidences[i] = evidences[i] * dir_fuse_weight[:, i:i + 1]
            evidence_a = evidence_a + evidences[i]'''
        return evidences, evidence_a#, dir_fuse_weight, self.e_parameters.detach()

#MLP 每个模态一个MLP
class BaseMLP(nn.Module):
    def __init__(self, num_views, dims, num_classes):
        super(BaseMLP, self).__init__()
        self.name = 'BaseMLP'
        #self.atten = atten
        self.num_views = num_views
        self.num_classes = num_classes
        self.num_layers = len(dims)
        #self.atten_para = nn.Parameter(torch.ones(num_views, dims[0][0]))
        #self.EvidenceCollectors = nn.ModuleList(
            #[EvidenceCollector(dims[i], self.num_classes) for i in range(self.num_views)])
        self.fc1 = nn.Linear(dims[self.num_layers - 1][0], 256)  # 第一层
        self.fc2 = nn.Linear(256,  num_classes)  # 输出层
        self.relu = nn.ReLU()


    def forward(self, X):
        # get evidence
        evidences = dict()
        for v in range(self.num_views):
            fc1 = self.fc1(X[v])  # 前向传播到第一层
            fc1_relu = self.relu(fc1)  # 激活函数
            evidences[v] = self.fc2(fc1_relu)  # 前向传播到输出层
            #evidences[v] = self.EvidenceCollectors[v](X[v])

        evidence_a = evidences[0]

        for i in range(1, self.num_views):
            evidence_a += evidences[i]  # 不对的，但是后面会有get_ea修正
        return evidences,evidence_a

# 所有模态共享一个MLP
class BaseMLP_Share(nn.Module):
    def __init__(self, num_views, dims, num_classes):
        super(BaseMLP_Share, self).__init__()
        self.name = 'BaseMLP_Share'
        self.num_layers = len(dims)
        self.num_views = num_views
        #self.atten_para = nn.Parameter(torch.ones(dims[0][0]))
        self.num_classes = num_classes
        self.EvidenceCollector = EvidenceCollector(dims[0], self.num_classes)
        self.fc1 = nn.Linear(dims[self.num_layers - 1][0], 256)  # 第一层
        self.fc2 = nn.Linear(256, num_classes)  # 输出层
        self.relu = nn.ReLU()


    def forward(self, X):

        # get evidence

        #for v in range(self.num_views):
            #evidence = self.EvidenceCollector(X)
        fc1 = self.fc1(X)  # 前向传播到第一层
        fc1_relu = self.relu(fc1)
        evidence = self.fc2(fc1_relu)

        return evidence #

def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha).cuda()
    dg1 = torch.digamma(alpha).cuda()
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p.long(), num_classes=torch.tensor(c, dtype=torch.int64))
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)

    alp = (E * (1 - label) + 1).cuda()
    B = annealing_coef * KL(alp, c)

    return (A + B)

def mse_loss(p, alpha, c, global_step, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    return (A + B) + C


class TMC(nn.Module):
    # model(num_views, dims, num_classes,atten)
    # model(X,Y['syn'],global_step=epoch+1)
    def __init__(self, views, classifier_dims,classes, lambda_epochs=50):
        """
        :param classes: Number of classification categories
        :param views: Number of views
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(TMC, self).__init__()
        self.name = 'TMC'
        self.views = views
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        self.Classifiers = nn.ModuleList([Classifier(classifier_dims[i], self.classes) for i in range(self.views)])

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = self.classes/S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha)-1):
            if v==0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a

    def forward(self, X, y, global_step):
        evidence = self.infer(X)
        loss = 0
        alpha = dict()
        for v_num in range(self.views):
            alpha[v_num] = evidence[v_num] + 1
            loss += ce_loss(y, alpha[v_num], self.classes, global_step=global_step, annealing_step=50)
        alpha_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1
        loss += ce_loss(y, alpha_a, self.classes, global_step, self.lambda_epochs)
        loss = torch.mean(loss)
        return evidence, evidence_a, loss

    def infer(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        evidence = dict()
        for v_num in range(self.views):
            evidence[v_num] = self.Classifiers[v_num](input[v_num])
        return evidence


class Classifier(nn.Module):
    def __init__(self, classifier_dims, classes):
        super(Classifier, self).__init__()
        self.num_layers = len(classifier_dims)
        self.fc = nn.ModuleList()
        for i in range(self.num_layers-1):
            self.fc.append(nn.Linear(classifier_dims[i], classifier_dims[i+1]))
        self.fc.append(nn.Linear(classifier_dims[self.num_layers-1], classes))
        self.fc.append(nn.Softplus())

    def forward(self, x):
        h = self.fc[0](x)
        for i in range(1, len(self.fc)):
            h = self.fc[i](h)
        return h
