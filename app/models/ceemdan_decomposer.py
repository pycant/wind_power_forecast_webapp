

import pandas as pd
from pyemd import CEEMDAN, EMD
import numpy as np
import matplotlib.pyplot as plt


def run_ceemdan_decomposition(file_path,output_path):
    # 设置绘图参数
    plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})

    try:
        df = pd.read_csv(file_path, encoding='utf-8').head(100)  # 尝试使用 GBK 编码
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin-1').head(100) # 尝试使用 Latin-1 编码

    # 提取完整数据
    t = df.index  # 时间索引
    s = df['Power (MW)'].values  # 提取'Power (MW)'列的数据

    # 创建CEEMDAN对象并进行分解
    ceemdan = CEEMDAN()
    IMFs_ceemdan = ceemdan(s)

    # 创建EMD对象并进行分解
    emd = EMD()
    IMFs_emd = emd.emd(s)  # 使用 emd.emd() 方法进行分解

    # 计算CEEMDAN和EMD的能量占比
    total_energy = np.sum(s**2)
    energies_ceemdan = [np.sum(imf**2) / total_energy for imf in IMFs_ceemdan]
    energies_emd = [np.sum(imf**2) / total_energy for imf in IMFs_emd]

    # 打印能量占比
    print("CEEMDAN Energy Ratios:")
    for i, energy in enumerate(energies_ceemdan, start=1):
        print(f'IMF {i}: {energy:.4f}')

    print("\nEMD Energy Ratios:")
    for i, energy in enumerate(energies_emd, start=1):
        print(f'IMF {i}: {energy:.4f}')

    # 创建一个新的DataFrame来保存CEEMDAN分解结果
    IMF_ceemdan_df = pd.DataFrame(IMFs_ceemdan.T, index=t, columns=[f'IMF {i+1}' for i in range(len(IMFs_ceemdan))])

    # 将原始信号添加到DataFrame中
    IMF_ceemdan_df['Original Signal'] = s

    # 保存CEEMDAN分解结果到新的Excel文件
    output_file_path = output_path
    IMF_ceemdan_df.to_excel(output_file_path)

    print(f"Decomposed data saved to {output_file_path}")

    # 设置绘图参数
    plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})
    plt.figure(figsize=(12, 18))

    # 绘制IMFs
    for i, imf in enumerate(IMFs_ceemdan):
        plt.subplot(len(IMFs_ceemdan) + 1, 1, i + 1)
        plt.plot(t, imf, color='blue', linewidth=1.2)
  
  
print("开始CEEMDAN分解...")
run_ceemdan_decomposition(r'data\cleaned\Wind farm site 1 (Nominal capacity-99MW)1746191929.0945573.csv',r'output/ceemdan_decomposition.csv')
