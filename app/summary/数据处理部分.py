#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = r"C:\Users\chenjin\Downloads\clustering_results.csv"
df = pd.read_csv(file_path)

# 提取'Power (kW)'列的数据
power_data = df['Power (MW)']

# 设置图像大小，根据数据量调整
plt.figure(figsize=(15, 6))  # 可以根据需要调整图像的宽度和高度

# 绘制折线图
plt.plot(power_data, label='Power (kW)', color='blue', linewidth=1.0)

# 添加标题和标签
plt.title('Power (kW) Over Time')
plt.xlabel('Time Index')
plt.ylabel('Power (kW)')

# 添加图例
plt.legend()

# 优化x轴标签显示，避免重叠
plt.xticks(rotation=45)

# 显示网格
plt.grid(True)

# 显示图像
plt.tight_layout()  # 自动调整子图参数，以确保子图之间有足够空间
plt.show()


# In[10]:


import pandas as pd
from statsmodels.tsa.stattools import adfuller

# 读取CSV文件
file_path = r"C:\Users\chenjin\Downloads\clustering_results.csv"
df = pd.read_csv(file_path)

# 提取'Power (kW)'列的数据
power_data = df['Power (MW)']

# 进行ADF检验
result = adfuller(power_data)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('t%s: %.3f' % (key, value))

# 根据p值判断平稳性
if result[1] < 0.05:
    print("序列是平稳的")
else:
    print("序列是非平稳的")


# In[19]:


import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = r"C:\Users\chenjin\Downloads\clustering_results.csv"
df = pd.read_csv(file_path)

# 提取'Power (MW)'和'Wind speed - at the height of wheel hub (m/s)'列的数据
power_mw = df['Power (MW)']
wind_speed_ms = df['Wind speed - at the height of wheel hub (m/s)']

# 设置图像大小，根据数据量调整
plt.figure(figsize=(12, 8))  # 可以根据需要调整图像的宽度和高度

# 绘制散点图，减小点的大小
plt.scatter(wind_speed_ms, power_mw, label='Power vs Wind Speed', color='blue', s=1, alpha=0.6)

# 添加标题和标签
plt.title('Scatter Plot of Power (MW) vs Wind Speed (m/s)')
plt.xlabel('Wind Speed at the Height of Wheel Hub (m/s)')
plt.ylabel('Power (MW)')

# 添加图例
plt.legend()

# 显示网格
plt.grid(True)

# 优化x轴和y轴标签显示，避免重叠
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# 显示图像
plt.tight_layout()  # 自动调整子图参数，以确保子图之间有足够空间
plt.show()


# In[ ]:





# In[24]:


import pandas as pd
from PyEMD import CEEMDAN, EMD
import numpy as np
import matplotlib.pyplot as plt

# 设置绘图参数
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})

# 读取CSV文件
file_path = r"C:\Users\chenjin\Downloads\clustering_results.csv"
df = pd.read_csv(file_path)  # 读取CSV文件

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
output_file_path = r"C:\Users\chenjin\Downloads\IMFs_decomposed_ceemdan_full.xlsx"
IMF_ceemdan_df.to_excel(output_file_path)

print(f"Decomposed data saved to {output_file_path}")

# 绘制EMD分解结果
plt.figure(figsize=(12, 18))
for i, imf in enumerate(IMFs_emd):
    plt.subplot(len(IMFs_emd) + 1, 1, i + 1)
    plt.plot(t, imf, color='blue', linewidth=1.2)
    plt.title(f'IMF {i+1} (EMD)', fontsize=10)
    plt.xticks([]) if i != len(IMFs_emd) - 1 else plt.xlabel('Time', fontsize=10)
    plt.yticks([])

plt.subplot(len(IMFs_emd) + 1, 1, len(IMFs_emd) + 1)
plt.plot(t, s, color='black', linewidth=1.2, label='Original Signal')
plt.title('Original Signal (Power (MW))', fontsize=10)
plt.ylabel('Power (MW)', fontsize=10)
plt.legend(loc='upper right')
plt.tight_layout(pad=2.0)
plt.show()


# In[25]:


# 保存CEEMDAN分解结果到新的Excel文件
output_file_path = r"C:\Users\chenjin\Downloads\IMFs_decomposed_ceemdan_full.xlsx"
IMF_ceemdan_df.to_excel(output_file_path)

print(f"Decomposed data saved to {output_file_path}")


# In[ ]:





# In[30]:


# 设置绘图参数
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})
plt.figure(figsize=(12, 18))

# 绘制IMFs
for i, imf in enumerate(IMFs_ceemdan):
    plt.subplot(len(IMFs_ceemdan) + 1, 1, i + 1)
    plt.plot(t, imf, color='blue', linewidth=1.2)
    plt.title(f'IMF {i+1}', fontsize=10)
    plt.xticks([])
    plt.yticks([])

# 绘制原始信号
plt.subplot(len(IMFs_ceemdan) + 1, 1, len(IMFs_ceemdan) + 1)
plt.plot(t, s, color='black', linewidth=1.2, label='Original Signal')
plt.title('Original Signal (Power (MW))', fontsize=10)
plt.xlabel('Time', fontsize=10)
plt.ylabel('Power (MW)', fontsize=10)
plt.legend(loc='upper right')

# 调整布局
plt.tight_layout(pad=2.0)

# 显示图表
plt.show()


# In[28]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 假设 IMFs_ceemdan 是 CEEMDAN 分解的结果，s 是原始信号
# IMFs_ceemdan 是一个列表，每个元素是一个 IMF（形状为 (n_samples,)）

# 初始化相关系数列表
relations = []

# 计算每个 IMF 与原始信号的相关系数
for imf in IMFs_ceemdan:
    correlation = np.corrcoef(imf.flatten(), s)[0, 1]
    relations.append(correlation)

# 创建一个 DataFrame 来存储 IMF 编号和相关系数
relation_df = pd.DataFrame({
    'IMF': [f'IMF {i+1}' for i in range(len(IMFs_ceemdan))],  # IMF 编号从 1 开始
    'Correlation': relations
})

# 设置绘图风格
sns.set(style="whitegrid")

# 绘制热力图
plt.figure(figsize=(10, 6))
sns.heatmap(relation_df.set_index('IMF'), annot=True, cmap='coolwarm', cbar=True, fmt=".2f")
plt.title('Correlation Coefficient Heatmap between CEEMDAN IMFs and Power (MW)')
plt.xlabel('IMF')
plt.ylabel('Power (MW)')
plt.show()

# 绘制柱状图
plt.figure(figsize=(10, 6))
sns.barplot(x=relation_df['IMF'], y=relation_df['Correlation'], palette='coolwarm')
plt.title('Correlation Coefficient Barplot between CEEMDAN IMFs and Power (MW)')
plt.xlabel('IMF')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45)  # 旋转 x 轴标签，避免重叠
plt.show()


# In[14]:


plt.figure(figsize=(10, 6))
sns.barplot(x=relation_df['IMF'], y=relation_df['Correlation'], palette='coolwarm')
plt.title('Correlation Coefficient Barplot between IMFs and Power (MW)')
plt.xlabel('IMF')
plt.ylabel('Correlation Coefficient')
plt.show()


# In[ ]:




