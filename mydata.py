import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

# 数据处理

#连续变量 用归一化
def lianxu(x, min=0):
    x.fillna(x.mean(), inplace=True)  # 用均值填充空值
    arr_x = np.array(x).reshape(-1, 1)
    if min == 0:
        scaler = MinMaxScaler((0, 1))  # 归一化，(0, 1)为映射区间
    else:  # min=-1
        scaler = MinMaxScaler((-1, 1))
    # nor_arr_x = np.squeeze(scaler.fit_transform(arr_x))
    nor_arr_x = scaler.fit_transform(arr_x)
    # x.replace(to_replace=x.values, value=nor_arr_x,inplace=True)
    return nor_arr_x

#离散变量 用独热编码
def lisan(x, x_name):
    oh = pd.get_dummies(x, prefix=x_name)  # 独热编码, 空值全为0
    return oh


def add_lx(org, arr, lst):
    for column in lst:
        x = org[column]
        nor_x = lianxu(x)
        arr = np.concatenate((arr, nor_x), axis=1)

    return arr


def add_ls(org,arr, lst):
    for column in lst:
        x = org[column]
        nor_x = lisan(x, column)
        arr = np.concatenate((arr, nor_x), axis=1)  # 6 15 21 26 37 43
    return arr


def seperate_XY(XY, num):
    data_X = XY.iloc[:, num:].values  # 变成了array
    data_Y = XY.iloc[:, :num].values
    data_Y = data_Y.T
    return data_X, data_Y


def data_process(org,before = False):
    # 12个X变量
    bol_list = ['is_ct']
    lx_list = ['er_value', 'pr_value', 'ki67_value', 'age', 'rs_21_gene']
    ls_list = ['pathological_type', 'cerbb_2', 'histological_grade', 'tumor_phase', 'node_phase', 'her2_fish']
    # 没有剔除了rs_21_gene、her2_fish两个缺失值较多的变量
    #org[bol_list].fillna(-1, inplace=True)  #  布尔变量为空的用-1代替
    org[bol_list] = org[bol_list].fillna(-1).copy()
    tol = org[bol_list].values
    #tol = org['er_value'].values.reshape(-1,1)
    tol = add_lx(org, tol, lx_list)
    tol = add_ls(org, tol, ls_list)
    X = pd.DataFrame(tol)#.iloc[:,1:]
    X.index = org.index

    # 选的讨论之后每个医生的决策
    Y_list = ['chemotherapy_id_user_27','chemotherapy_id_user_28', 'chemotherapy_id_user_29',
              'chemotherapy_id_user_30', 'chemotherapy_id_user_31', 'chemotherapy_id_user_34']   #选了几个空值较少的user
    usr = len(Y_list)+1
    Y = org[Y_list]
    #Y.dropna(inplace=True)  # 删去有空值的行
    Y = Y.dropna().copy()
    #Y[Y > 18] = 19
    Y = pd.merge(Y, org['chemotherapy_id'], left_index=True, right_index=True)
    #Y.dropna(inplace=True)
    Y = Y.dropna().copy()
    Y.drop(Y[(Y == 0).any(axis=1)].index, inplace=True)  # 删去存在Y==0的行
    #Y.drop(Y.iloc[:-1][(Y.iloc[:-1] == 0).any(axis=1)].index, inplace=True)

    lst = Y['chemotherapy_id'].unique()
    Y.applymap(lambda x: x if x in lst else -1)  # 不在最终方案的令为-1
    Y = Y.astype(dtype=np.int64)

    XY = pd.merge(Y, X, left_index=True, right_index=True)
    # print(XY)
    ##XY.dropna(subset=XY.columns[usr], inplace=True)  # 删去'is_ct'为空的
    data_X, data_Y = seperate_XY(XY, usr)

    if before == True:   # 每个模态没有指向自己的Y
        return MultiViewDataset1("CT", data_X, data_Y,lst)
    else:  # 每个模态指向自己的Y
        return MultiViewDataset("CT", data_X, data_Y, XY)
    #return data_X, data_Y


def CT(before=False):
    org = pd.read_csv('data/R2V_MainUsers1.csv', encoding="utf-8", index_col=12).copy()
    org.drop('Unnamed: 0', axis=1, inplace=True)
    return data_process(org, before)


def CT1(new_data):
    org = pd.read_csv('data/R2V_MainUsers1.csv', encoding="utf-8", index_col=12).copy()
    org.drop('Unnamed: 0', axis=1, inplace=True)
    #new_data = new_data.replace(0, -1).copy()
    new_data.index = [org.index[-1] + 1]

    for col in org.columns:
        if col not in new_data.columns:
            new_data[col] = 1
            new_data = new_data.copy()

    #data = pd.concat([org, new_data], ignore_index=True).fillna(100)
    data = pd.concat([org, new_data])
    return data_process(data)

import scipy.io as sio

def PIE():
    # dims of views: 484 256 279
    print('PIE')
    data_path = "data/PIE_face_10.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['gt']
    for v in range(len(data_X)):
        data_X[v] = data_X[v].T
    return MultiViewDataset_PIE("PIE", data_X, data_Y)

def HandWritten():
    # dims of views: 240 76 216 47 64 6
    data_path = "data/handwritten.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['Y']
    return MultiViewDataset_HR("HandWritten", data_X, data_Y)

class MultiViewDataset(Dataset):
    def __init__(self, data_name, data_X, data_Y, XY):
        super(MultiViewDataset, self).__init__()
        self.data_name = data_name

        self.X = dict()
        self.Y = dict()  # !!
        self.num_views = len(data_Y) - 1
        for v in range(self.num_views):
            self.X[v] = self.normalize(data_X)
        self.X['index'] = XY.index.values
        for v in range(self.num_views):
            y_labels = data_Y[v]
            # if np.min(y_labels) == 1:
            #      y_labels = y_labels - 1
            y_labels = y_labels.astype(dtype=np.int64)
            self.Y[v] = y_labels.copy()

        self.Y['syn'] = data_Y[-1].astype(dtype=np.int64).copy()
        '''views_count = [0, 0, 0, 0, 0, 0]
        for i in range(6):
            for j in range(len(self.Y[i])):
                if self.Y[i][j] == self.Y['syn'][j]:
                    views_count[i] += 1

        arr = np.array(views_count).reshape(-1, 1)
        scaler = MinMaxScaler((0, 1))  # 归一化，(0, 1)为映射区间'''
        #poster = scaler.fit_transform(arr)
        #poster = nn.functional.softmax(torch.FloatTensor(views_count))


        # self.num_classes = len(np.unique(data_Y))
        self.num_classes = (data_Y[0].max().astype(dtype=np.int64)) + 1
        self.dims = self.get_dims()  # 每个模态维度

    def __getitem__(self, index):
        data = dict()
        target = dict()  # !!
        for v_num in range(self.num_views):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
            target[v_num] = self.Y[v_num][index]  # !!
        data['index'] = self.X['index'][index]
        target['syn'] = self.Y['syn'][index]  # !!  target有0 1 2 syn四个key
        return data, target, index

    def __len__(self):
        return len(self.X[0])

    def get_dims(self):
        dims = []
        for view in range(self.num_views):
            dims.append([self.X[view].shape[1]])
        return np.array(dims)

    @staticmethod
    def normalize(x, min=0):
        if min == 0:
            scaler = MinMaxScaler((0, 1))
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

    def postprocessing(self, index, addNoise=False, sigma=0, ratio_noise=0.5, addConflict=False,
                       ratio_conflicts=[0.1, 0.4, 0.4],view=None):
        if addNoise:
            self.addNoise(index, ratio_noise, sigma=sigma,view=view)
        if addConflict:
            # self.addConflict(index, ratio_conflict)
            for view, ratio_conflict in enumerate(ratio_conflicts):  # 0:0.1，1:0.4，2:0.4
                self.addConflict2View(index, ratio_conflict, view)
        pass

    def addNoise(self, index, ratio, sigma, view=None):
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            views = np.random.choice(np.array(self.num_views), size=np.random.randint(self.num_views), replace=False)
            if view==None:
                for v in views:
                    self.X[v][i] = np.random.normal(self.X[v][i], sigma)
            else:
                self.X[view][i] = np.random.normal(self.X[view][i], sigma)
        pass

    def addConflict2View(self, index, ratio, view):  # ！！
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            # self.Y[view][i] = (self.Y[view][i] + 1) % self.num_classes
            self.Y[view][i] = np.random.randint(self.num_classes)
            #print('self.Y[view][i] - ', self.Y[view][i])
            # print('self.Y[view][i] - syn', self.Y['syn'][i])
            # print('self.Y[view][i] - 2', self.Y[2][i])
        pass

class MultiViewDataset1(Dataset):
    def __init__(self, data_name, data_X, data_Y,lst):
        super(MultiViewDataset1, self).__init__()
        self.data_name = data_name
        self.Y_lst = lst
        self.X = dict()
        self.num_views = len(data_Y) - 1
        for v in range(self.num_views):
            self.X[v] = self.normalize(data_X)

        self.Y = data_Y[-1].astype(dtype=np.int64).copy()
        self.num_classes = (data_Y[0].max().astype(dtype=np.int64)) + 1
        self.dims = self.get_dims()

    def __getitem__(self, index):
        data = dict()
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
        target = self.Y[index]
        return data, target, index

    def __len__(self):
        return len(self.X[0])

    def get_dims(self):
        dims = []
        for view in range(self.num_views):
            dims.append([self.X[view].shape[1]])
        return np.array(dims)

    @staticmethod
    def normalize(x, min=0):
        if min == 0:
            scaler = MinMaxScaler((0, 1)) #归一化，(0, 1)为映射区间
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

    def postprocessing(self, index, addNoise=False, sigma=0, ratio_noise=0.5, addConflict=False, ratio_conflict=0.5):
        if addNoise:
            self.addNoise(index, ratio_noise, sigma=sigma)
        if addConflict:
            self.addConflict(index, ratio_conflict)
        pass

    def addNoise(self, index, ratio, sigma):
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            views = np.random.choice(np.array(self.num_views), size=np.random.randint(self.num_views), replace=False)
            for v in views:
                self.X[v][i] = np.random.normal(self.X[v][i], sigma)  #生成正态分布随机数，以self.X[v][i]为中心 标准差sigma
        pass

    def addConflict(self, index, ratio):
        records = dict()
        for c in range(1, self.num_classes):
            if c in self.Y_lst:
                i = np.where(self.Y == c)[0][0]  #取得每个类第一个的index
                temp = dict()
                for v in range(self.num_views):
                    temp[v] = self.X[v][i]  #取出第i个实例的三个view
            records[c] = temp  #记录指向类别c的实例
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)  #选40%的test
        for i in selects:
            v = np.random.randint(self.num_views) #生成随机整数
            self.X[v][i] = records[(self.Y[i] + 1) % self.num_classes][v]  #改模态最终指向的Y+1
        pass

class MultiViewDataset_PIE(Dataset):
    def __init__(self, data_name, data_X, data_Y):
        super(MultiViewDataset_PIE, self).__init__()
        self.data_name = data_name

        self.X = dict()
        self.Y = dict()  # !!
        self.num_views = data_X.shape[0]
        for v in range(self.num_views):
            self.X[v] = self.normalize(data_X[v])

        y_labels = np.squeeze(data_Y)  # !! 删除单维度条目
        if np.min(y_labels) == 1:
            y_labels = y_labels - 1
        y_labels = y_labels.astype(dtype=np.int64)
        for v in range(self.num_views):  # !!
            self.Y[v] = y_labels.copy()  # !!  这里每个self.Y[v]都是一样的，需要加冲突
        self.Y['syn'] = y_labels.copy()  # !!
        self.Y['syn'].mean()

        self.num_classes = len(np.unique(y_labels))
        self.dims = self.get_dims()

    def __getitem__(self, index):
        data = dict()
        target = dict()  # !!
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
            target[v_num] = self.Y[v_num][index]  # !!
        target['syn'] = self.Y['syn'][index]  # !!  target有0 1 2 syn四个key
        return data, target, index

    def __len__(self):
        return len(self.X[0])

    def get_dims(self):
        dims = []
        for view in range(self.num_views):
            dims.append([self.X[view].shape[1]])
        return np.array(dims)

    @staticmethod
    def normalize(x, min=0):
        if min == 0:
            scaler = MinMaxScaler((0, 1))
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

    def postprocessing(self, index, addNoise=False, sigma=0, ratio_noise=0.5, addConflict=False,
                       ratio_conflicts=[0.1, 0.4, 0.4],ratio_changes=[0.5,0.5,0.5]):
        if addNoise:
            self.addNoise(index, ratio_noise, sigma=sigma)
        if addConflict:  # 加冲突
            # self.addConflict(index, ratio_conflict)
            #for view in range(self.num_views):
            for view, ratio_change in enumerate(ratio_changes):
                self.addConflict3View(view, ratio_change)
            for view, ratio_conflict in enumerate(ratio_conflicts):  # 0:0.1，1:0.4，2:0.4,每个view的冲突度不一样
                self.addConflict2View(index, ratio_conflict, view)
        pass

    def addNoise(self, index, ratio, sigma):
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            views = np.random.choice(np.array(self.num_views), size=np.random.randint(self.num_views), replace=False)
            for v in views:
                self.X[v][i] = np.random.normal(self.X[v][i], sigma)
        pass

    def addConflict(self, index, ratio):
        records = dict()
        for c in range(self.num_classes):
            i = np.where(self.Y == c)[0][0]
            temp = dict()
            for v in range(self.num_views):
                temp[v] = self.X[v][i]
            records[c] = temp
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            v = np.random.randint(self.num_views)
            self.X[v][i] = records[(self.Y[i] + 1) % self.num_classes][v]
        pass

    def addConflict2View(self, index, ratio, view):  # ！！
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)  # 在index里面随机选一些数
        for i in selects:
            # print('view:', view,'i:',i)
            # print('self.Y[view][i]', self.Y[view][i])
            # self.Y[view][i] = (self.Y[view][i] + 1) % self.num_classes
            self.Y[view][i] = np.random.randint(self.num_classes)   #第i个样本对应的view模态的Y，随机变换
            # print('self.Y[view][i] - ', self.Y[view][i])
            # print('self.Y[view][i] - syn', self.Y['syn'][i])
            # print('self.Y[view][i] - 2', self.Y[2][i])
        pass

    def addConflict3View(self, view, ratio, ratio_all=2.4):
        Xview_mean = self.X[view].mean(axis=1)  # 得2000个数
        # 求均值和方差
        mean, std = np.mean(Xview_mean), np.std(Xview_mean)
        for i in range(len(Xview_mean)):
            if Xview_mean[i] > (mean + ratio * std) or i < (mean - ratio * std):  # 在ratio西格玛外
                self.Y[view][i] = np.random.randint(self.num_classes)
            if Xview_mean[i] > (mean + 2 * ratio * std) or i < (mean - 2 * ratio * std):  # 在1.5ratio西格玛外
                self.Y[self.num_views -1][i] = np.random.randint(self.num_classes) # 代表主任
            #if Xview_mean[i] > (mean + ratio_all * std) or i < (mean - ratio_all * std):  # 在2西格玛外
                #self.Y['syn'][i] = np.random.randint(self.num_classes)
        pass

class MultiViewDataset_HR(Dataset):
    def __init__(self, data_name, data_X, data_Y):
        super(MultiViewDataset_HR, self).__init__()
        self.data_name = data_name

        self.X = dict()
        self.Y = dict()  # !!
        self.num_views = data_X.shape[0]
        for v in range(self.num_views):
            self.X[v] = self.normalize(data_X[v])

        y_labels = np.un(data_Y)  # !! 删除单维度条目
        if np.min(y_labels) == 1:
            y_labels = y_labels - 1
        y_labels = y_labels.astype(dtype=np.int64)
        for v in range(self.num_views):  # !!
            self.Y[v] = y_labels.copy()  # !!  这里每个self.Y[v]都是一样的，需要加冲突
        self.Y['syn'] = y_labels.copy()  # !!
        self.Y['syn'].mean()

        self.num_classes = len(np.unique(y_labels))
        self.dims = self.get_dims()

    def __getitem__(self, index):
        data = dict()
        target = dict()  # !!
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
            target[v_num] = self.Y[v_num][index]  # !!
        target['syn'] = self.Y['syn'][index]  # !!  target有0 1 2 syn四个key
        return data, target, index

    def __len__(self):
        return len(self.X[0])

    def get_dims(self):
        dims = []
        for view in range(self.num_views):
            dims.append([self.X[view].shape[1]])
        return np.array(dims)

    @staticmethod
    def normalize(x, min=0):
        if min == 0:
            scaler = MinMaxScaler((0, 1))
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

    def postprocessing(self, index, addNoise=False, sigma=0, ratio_noise=0.5, addConflict=False,
                       ratio_conflicts=[0.1, 0.4, 0.4],ratio_changes=[0.2,0.265,0.275,0.285,0.295,0.305]):
        if addNoise:
            self.addNoise(index, ratio_noise, sigma=sigma)
        if addConflict:  # 加冲突
            # self.addConflict(index, ratio_conflict)
            #for view in range(self.num_views):
            for view, ratio_change in enumerate(ratio_changes):
                self.addConflict4View(view, ratio_change,index)
            #for view, ratio_conflict in enumerate(ratio_conflicts):  # 0:0.1，1:0.4，2:0.4,每个view的冲突度不一样
                #self.addConflict2View(index, ratio_conflict, view)
        pass

    def addNoise(self, index, ratio, sigma):
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            views = np.random.choice(np.array(self.num_views), size=np.random.randint(self.num_views), replace=False)
            for v in views:
                self.X[v][i] = np.random.normal(self.X[v][i], sigma)
        pass

    def addConflict(self, index, ratio):
        records = dict()
        for c in range(self.num_classes):
            i = np.where(self.Y == c)[0][0]
            temp = dict()
            for v in range(self.num_views):
                temp[v] = self.X[v][i]
            records[c] = temp
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            v = np.random.randint(self.num_views)
            self.X[v][i] = records[(self.Y[i] + 1) % self.num_classes][v]
        pass

    def addConflict2View(self, index, ratio, view):  # ！！
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)  # 在index里面随机选一些数
        for i in selects:
            # print('view:', view,'i:',i)
            # print('self.Y[view][i]', self.Y[view][i])
            # self.Y[view][i] = (self.Y[view][i] + 1) % self.num_classes
            self.Y[view][i] = np.random.randint(self.num_classes)   #第i个样本对应的view模态的Y，随机变换
            # print('self.Y[view][i] - ', self.Y[view][i])
            # print('self.Y[view][i] - syn', self.Y['syn'][i])
            # print('self.Y[view][i] - 2', self.Y[2][i])
        pass

    def addConflict3View(self, view, ratio, ratio_all=2.4):
        Xview_mean = self.X[view].mean(axis=0)
        Xview_mean1 = self.X[view].mean(axis=1) # 得2000个数
        # 求每个模态的均值和方差
        mean, std = np.mean(Xview_mean), np.std(Xview_mean)
        for i in range(len(Xview_mean1)):
            if( Xview_mean1[i] > (mean + ratio * std)) or (Xview_mean1[i] < (mean - ratio * std)):  # 在ratio西格玛外
                self.Y[view][i] = np.random.randint(self.num_classes)
            if (Xview_mean1[i] > (mean + 3 * ratio * std)) or  (Xview_mean1[i] < (mean - 3 * ratio * std)):  # 在1.5ratio西格玛外
                self.Y[self.num_views -1][i] = np.random.randint(self.num_classes) # 代表主任
            #if Xview_mean[i] > (mean + ratio_all * std) or i < (mean - ratio_all * std):  # 在2西格玛外
                #self.Y['syn'][i] = np.random.randint(self.num_classes)

    def addConflict4View(self, view, ratio, index):
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)  # 选train_index里面17%的序号
        for i in selects:
            self.Y[view][i] = np.random.randint(self.num_classes)

        pass