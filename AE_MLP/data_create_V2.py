# This file is to generate train dataset and validation dataset. This file should run first.


# Before running this file, please create a file called "dataset" and "train_val_data"
# "dataset" is to put the collected data in it.
# "train_val_data" is to store the generated train and val data
# There is no input is required.

# This file can generate a dataset with name form: 0a_train_set.csv
# 0:
#	prepocess mode for RSS data. 
#		0: no preprocess is applied
#		1: Delete the APs which is always -100 and which appears in Phone A's dataset and Phone B's dataset.
#		2: For Phone A's data, delete all APs that are always -110. For Phone B, it is similar
# a:
#	mobile phone type
#		a: Phone A, which is MIX2.
#		b: Phone B, whihc is HuaWei.
#		ab: Combination of Phone A and Phone B
# train_set:
#	train data or val data.


import csv
import pandas  as pd
import numpy as np
import sys
import torch
import random
import os
import matplotlib.pyplot as plt

sample_num = 800
rate = 0.25
delete = ['0', '1', '2']

# This function is to open the original data depending on phone combinations.
def file_open(input_mode):
    # 确定数据集的类型
    train_mode, val_mode = input_mode.split('-', 1)

    file_a = 'dataset/Dataset_4thFloor_MIX2.csv'  # Is_file_a=input_mode.find('a')
    file_b = 'dataset/Dataset_4thFloor_HuaWei.csv'  # Is_file_b=input_mode.find('b')

    if input_mode.find('a') != -1:
        data_a = pd.read_csv(file_a)
        data_a = data_a.iloc[:, 1:534].drop(
            columns=['Floor', 'Building', 'Model', 'GeoX', 'GeoY', 'GeoZ', 'AccX', 'AccY', 'AccZ', 'OriX', 'GriY',
                     'GriZ', ])  # loc: for label based indexing
    else:  # iloc:for positional indexing
        data_a = pd.DataFrame(None)
    if input_mode.find('b') != -1:
        data_b = pd.read_csv(file_b)
        data_b = data_b.iloc[:, 1:534].drop(
            columns=['Floor', 'Building', 'Model', 'GeoX', 'GeoY', 'GeoZ', 'AccX', 'AccY', 'AccZ', 'OriX', 'GriY',
                     'GriZ', ])
    else:
        data_b = pd.DataFrame(None)  # set an empty DataFrame
    mode = {'train_set': train_mode, 'val_set': val_mode}
    return mode, data_a, data_b

# This function is to save the train and val data separately.
def data_save(mode, D, data_a, data_b):
    index = {'train_set': [], 'val_set': []}
    index['train_set'], index['val_set'] = train_val_index()  # generate random and unique index for train data and val data

    for t_v in mode.keys():
        data_set = select_data(index[t_v], mode[t_v], t_v, data_a, data_b)  # Based on index, select data from original data
        filename = 'train_val_data/' + D + mode[t_v] + '_' + t_v + '.csv'
        print(filename)
        print(data_set.shape)
        #data_set.to_csv(filename)

# This function is to generate random index without reputations
def random_generate(num, range_num):
    if num < 600:
        index = np.random.randint(range_num, size=num)
        unique_idx = np.unique(index)
        diff_idx = np.zeros(1)
        while diff_idx.shape[0] + unique_idx.shape[0] < num:
            sub_idx = np.unique(np.random.randint(range_num, size=num))
            diff_idx = np.setdiff1d(sub_idx, unique_idx)  # 在 unique_idx中查找不存在于sub_idx中的元素
        index = np.append(unique_idx, diff_idx[:num - unique_idx.shape[0]])
        index = index.tolist()
    else:
        index = np.random.randint(range_num, size=range_num - num)
        unique_idx = np.unique(index)
        diff_idx = np.zeros(1)
        while diff_idx.shape[0] + unique_idx.shape[0] < range_num - num:
            sub_idx = np.unique(np.random.randint(range_num, size=range_num - num))
            diff_idx = np.setdiff1d(sub_idx, unique_idx)  # 在 unique_idx中查找不存在于sub_idx中的元素
            if diff_idx.shape[0] + unique_idx.shape[0] > range_num - num:
                diff_idx = diff_idx[:range_num - num - unique_idx.shape[0]]
        index = np.append(unique_idx, diff_idx[:range_num - num - unique_idx.shape[0]])
        full_size = set(range(range_num))
        index = set(index)
        index = list(full_size - index)
    return index

# This function is to use "random_generate" to get train and validation index without identical data.
def train_val_index(num_train=sample_num,  range_num=1238):
    num_val = int(num_train * rate)
    index_train = random_generate(num_train, range_num)
    index_val = random_generate(num_val, range_num)
    index_common = np.zeros(1)
    while index_common.shape[0] < num_val:
        index_new = random_generate(num_val, range_num)
        index_new = np.setdiff1d(index_new, index_train)
        index_common = np.unique(np.append(index_common, index_new, axis=None))
    index_val = index_common[:num_val]
    return index_train, index_val

# This function is to select data of processed data-a and data-b based on train-validation or whether is ab-combination
def select_data(index, mode, t_v, data_a, data_b, number=sample_num):
    if t_v == 'val_set': number = int(number * rate)
    data_set = None
    if mode == 'a':
        data_set = data_a.loc[index]
    if mode == 'b':
        data_set = data_b.loc[index]
    if mode == 'ab':
        # If the phone combination is ab. The function will select a and b dataset with same index provided by upper function.
        # Then it will generate new indexes which is half of the desired data size.
        # Select the processed data of a and b based on new index. Then the combination of a and b will meet the desire size.
        ab_a = data_a.loc[index].reset_index(drop=True)  # reset_index is to reset the original index into a sequential from 1
        ab_b = data_b.loc[index].reset_index(drop=True)
        index, _ = train_val_index(num_train=int(number / 2), range_num=ab_a.shape[0])
        ab_a = ab_a.loc[index]
        index, _ = train_val_index(num_train=int(number / 2), range_num=ab_b.shape[0])
        ab_b = ab_b.loc[index]
        data_set = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
        # Use list to combine a and b data
        for column in ab_a.columns:
            combine = []
            combine.extend(ab_a[column])
            combine.extend(ab_b[column])
            data_set[column] = combine
    data_set = data_set.reset_index(drop=True)
    return data_set

# This function is to find the common columns in a and b data.
# The common columns mean this column in phone a and b are always -110.
def find_common(data_a, data_b):
    AP_a = []
    for key, val in data_a.iteritems():
        if key != 'Loc_x' or key != 'Loc_y':
            val = np.asarray(val)
            strength = np.unique(val)
            if strength.shape[0] == 1:
                AP_a.append(key)
    AP_a = set(AP_a)
    AP_b = []
    for key, val in data_b.iteritems():
        if key != 'Loc_x' or key != 'Loc_y':
            val = np.asarray(val)
            strength = np.unique(val)
            if strength.shape[0] == 1:
                AP_b.append(key)
    AP_b = set(AP_b)
    common = AP_b & AP_a
    # print(common)
    return common

# Delete common columns in the dataset
def delete_common(common, **kwargs):
    for data_set in kwargs:
        if not kwargs[data_set].empty:
            AP = []
            data = pd.DataFrame(None)
            for key, val in kwargs[data_set].iteritems():
                val = np.asarray(val)
                if key not in common or key == 'rGeoX':
                    data[key] = val
            kwargs[data_set] = data
    return kwargs.values()

# Delete all the columns that are -110 in a and b dataset.
# !!!!warning!!!!
    # This will result different size of a and b data.
    # The model or the data should be adjusted before training and combining
def delete_all(**kwargs):
    for data_set in kwargs:
        if not kwargs[data_set].empty:
            data = pd.DataFrame(None)
            for key, val in kwargs[data_set].iteritems():
                val = np.asarray(val)
                if np.unique(val).shape[0] != 1 or key == 'rGeoX':
                    data[key] = val
            kwargs[data_set] = data
    return kwargs.values()


if __name__ == "__main__":

    # common=['WAP028', 'WAP343', 'WAP053', 'WAP262', 'WAP441', 'WAP163', 'WAP478', 'WAP051', 'WAP166', 'WAP023', 'WAP461', 'WAP121', 'WAP348', 'WAP295', 'WAP489', 'WAP254', 'WAP085', 'WAP038', 'WAP132', 'WAP065', 'WAP288', 'WAP418', 'WAP444', 'WAP360', 'WAP300', 'WAP310', 'WAP152', 'WAP472', 'WAP167', 'WAP334', 'WAP329', 'WAP285', 'WAP427', 'WAP509', 'WAP078', 'WAP311', 'WAP045', 'WAP298', 'WAP074', 'WAP496', 'WAP415', 'WAP096', 'WAP368', 'WAP424', 'WAP391', 'WAP067', 'WAP035', 'WAP369', 'WAP005', 'WAP070', 'WAP366', 'WAP452', 'WAP426', 'WAP061', 'WAP501', 'WAP040', 'WAP104', 'WAP077', 'WAP127', 'WAP169', 'WAP208', 'WAP022', 'WAP217', 'WAP261', 'WAP266', 'WAP098', 'WAP289', 'WAP221', 'WAP056', 'WAP364', 'WAP443', 'WAP020', 'WAP282', 'WAP333', 'WAP377', 'WAP072', 'WAP026', 'WAP491', 'WAP183', 'WAP235', 'WAP207', 'WAP463', 'WAP174', 'WAP352', 'WAP430', 'WAP018', 'WAP321', 'WAP494', 'WAP097', 'WAP487', 'WAP313', 'WAP091', 'WAP361', 'WAP412', 'WAP508', 'WAP009', 'WAP405', 'WAP147', 'WAP317', 'WAP105', 'WAP231', 'WAP473', 'WAP081', 'WAP283', 'WAP148', 'WAP305', 'WAP192', 'WAP182', 'WAP144', 'WAP302', 'WAP375', 'WAP128', 'WAP205', 'WAP493', 'WAP512', 'WAP428', 'WAP117', 'WAP456', 'WAP153', 'WAP200', 'WAP420', 'WAP340', 'WAP393', 'WAP124', 'WAP146', 'WAP250', 'WAP359', 'WAP017', 'WAP000', 'WAP002', 'WAP019', 'WAP021', 'WAP357', 'WAP434', 'WAP100', 'WAP271', 'WAP015', 'WAP350', 'WAP226', 'WAP488', 'WAP378', 'WAP265', 'WAP211', 'WAP106', 'WAP392', 'WAP108', 'WAP006', 'WAP495', 'WAP161', 'WAP219', 'WAP178', 'WAP135', 'WAP059', 'WAP033', 'WAP247', 'WAP304', 'WAP276', 'WAP490', 'WAP419', 'WAP112', 'WAP356', 'WAP325', 'WAP209', 'WAP058', 'WAP315', 'WAP086', 'WAP229', 'WAP275', 'WAP216', 'WAP303', 'WAP312', 'WAP492', 'WAP291', 'WAP309', 'WAP475', 'WAP099', 'WAP355', 'WAP203', 'WAP374', 'WAP287', 'WAP462', 'WAP109', 'WAP080', 'WAP296', 'WAP252', 'WAP338', 'WAP141', 'WAP071', 'WAP239', 'WAP485']
    # common columns
    mode, data_a, data_b = file_open(input_mode='ab-ab')
    common = find_common(data_a, data_b)  # 找出A B 手机测量RSS都为-110的AP
    for D in delete:
        if D != '2':
            all_mode = ['a-a', 'b-b', 'ab-ab']
        else:
            all_mode = ['a-a', 'b-b']
        for input_mode in all_mode:
            mode, data_a, data_b = file_open(input_mode)  # 确定训练-测试模式 以及 数据集
            if D == '1':
                data_a, data_b = delete_common(common, data_a=data_a, data_b=data_b)
            if D == '2':
                # plotting_figure(common,data_a=data_a,data_b=data_b)
                data_a, data_b = delete_all(data_a=data_a, data_b=data_b)
            data_save(mode, D, data_a, data_b)
