# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:22:05 2024

@author: jmen
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import chardet
import numpy as np
from tqdm import tqdm

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def find_median_and_indices(arr):
    # 排除 NaN 值
    valid_data = arr[~np.isnan(arr)]
    
    # 对有效数据进行排序
    sorted_data = np.sort(valid_data)
    n = len(sorted_data)
    
    # 计算中位数
    median_value = np.median(sorted_data)

    # 如果是偶数个数，找到中间两个数
    if n % 2 == 0:
        lower_median_index = (n // 2) - 1  # 中间较小数的索引
        upper_median_index = n // 2        # 中间较大数的索引
        lower_value = sorted_data[lower_median_index]
        upper_value = sorted_data[upper_median_index]
        
        # 查找原数据中的索引（可能返回多个匹配的索引）
        indices = np.argwhere(np.isin(arr, [lower_value, upper_value])).flatten().tolist()
    else:
        # 如果是奇数个数，直接返回中位数和它的索引
        median_value = sorted_data[n // 2]
        indices = np.argwhere(np.isclose(arr, median_value)).flatten().tolist()

    return median_value, indices
# 设置输入文件夹路径
input_folder = r'C:\Users\jmen\Box\ERSL_FieldDatabase\LakeLanier\2024October7\YSI\Raw\EXO2'
output_name = r'LakeLanier_EXO2_20241007'
target_path = r'C:\Users\jmen\Box\ERSL_FieldDatabase\LakeLanier\2024October7\YSI\Median'
out_path = os.path.join(target_path,output_name+'_median.csv')

# 创建一个空列表存储数据
data = []

# 遍历输入文件夹中的所有CSV文件
for filename in tqdm(os.listdir(input_folder)):
    file_path = os.path.join(input_folder, filename)
    encoding = detect_encoding(file_path)
    if filename.endswith('.csv'):
        df = pd.read_csv(file_path, skiprows=17, index_col=False, encoding=encoding)
    elif filename.endswith('.xlsx'):
        df = pd.read_excel(file_path, skiprows=17, index_col=False, engine=encoding)
    
    chlorophyll = df['Chl ug/L'] 
    if len(chlorophyll)%2 == 0:
        median_index = np.argsort(chlorophyll)[len(chlorophyll)//2]
        median_index = np.array(median_index[...,np.newaxis])
        line_median = df.iloc[median_index,:]
        data.append(line_median)
    else:
        index_median = np.argwhere(chlorophyll == np.nanmedian(chlorophyll))
        if len(index_median) == 1:
            line_median = df.iloc[index_median[0],:]
            data.append(line_median)
        elif len(index_median) > 1:
            line_median = df.iloc[index_median[0],:]
            data.append(line_median)
        else:
            print('No median data')
            
# 合并所有数据到一个DataFrame
combined_df = pd.concat(data, ignore_index=True)

combined_df.to_csv(out_path)
