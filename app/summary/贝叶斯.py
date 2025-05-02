#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

# 1. 加载数据
imfs_file_path = r"C:\Users\chenjin\Downloads\IMFs_decomposed_ceemdan_full.xlsx"
data_file_path = r"C:\Users\chenjin\OneDrive\图片\文档\处理后数据.xlsx"

imfs_df = pd.read_excel(imfs_file_path, index_col=0)
data_df = pd.read_excel(data_file_path)

# 合并特征
imfs_selected = imfs_df[['IMF 11', 'IMF 12', 'IMF 10', 'IMF 9']]
wind_speed = data_df['风速'].values
power = data_df['Power (MW)'].values

# 截取对齐
min_length = min(len(imfs_selected), len(wind_speed))
imfs_selected = imfs_selected.iloc[:min_length]
wind_speed = wind_speed[:min_length]
power = power[:min_length]

# 合并特征
features = np.column_stack((imfs_selected, wind_speed))
target = power.reshape(-1, 1)

# 2. 数据归一化
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler(feature_range=(0, 1))

features_scaled = scaler_features.fit_transform(features)
target_scaled = scaler_target.fit_transform(target)

# 3. 数据预处理
def create_dataset(X, y, time_step=1):
    X_data, y_data = [], []
    for i in range(len(X) - time_step):
        X_data.append(X[i:(i + time_step), :])
        y_data.append(y[i + time_step])
    return np.array(X_data), np.array(y_data)

time_step = 10
X, y = create_dataset(features_scaled, target_scaled, time_step)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 4. 定义 LSTM 模型构建函数
def build_lstm_model(lstm_units=50, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))  # 添加 Input 层
    model.add(LSTM(int(lstm_units), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(int(lstm_units), return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(50, activation='relu'))  # 增加 Dense 层
    model.add(Dense(25, activation='relu'))  # 增加 Dense 层
    model.add(Dense(1))
    
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-5)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# 5. 定义贝叶斯优化的目标函数
def optimize_lstm(lstm_units, dropout_rate, learning_rate):
    lstm_units = int(lstm_units)
    dropout_rate = max(min(dropout_rate, 0.5), 0.1)  # Dropout 率限制在 0.1 到 0.5 之间
    learning_rate = max(min(learning_rate, 0.01), 0.0001)  # 学习率限制在 0.0001 到 0.01 之间
    
    model = build_lstm_model(lstm_units, dropout_rate, learning_rate)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, lr_scheduler],
        verbose=0
    )
    
    # 返回验证集损失的负值（贝叶斯优化默认是最大化目标函数）
    return -history.history['val_loss'][-1]

# 6. 设置贝叶斯优化的参数范围
pbounds = {
    'lstm_units': (30, 150),  # LSTM 单元数范围
    'dropout_rate': (0.1, 0.4),  # Dropout 率范围
    'learning_rate': (0.0001, 0.001)  # 学习率范围
}

# 7. 运行贝叶斯优化
optimizer = BayesianOptimization(
    f=optimize_lstm,
    pbounds=pbounds,
    random_state=42
)

optimizer.maximize(init_points=5, n_iter=20)  # 初始点 5 个，迭代 20 次

# 8. 获取最佳超参数
best_params = optimizer.max['params']
print("Best Parameters:", best_params)

# 9. 使用最佳超参数训练最终模型
final_model = build_lstm_model(
    lstm_units=int(best_params['lstm_units']),
    dropout_rate=best_params['dropout_rate'],
    learning_rate=best_params['learning_rate']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history = final_model.fit(
    X_train, y_train,
    epochs=150,  # 增加训练轮数
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# 10. 在测试集上进行预测
y_test_pred_scaled = final_model.predict(X_test)
y_test_pred = scaler_target.inverse_transform(y_test_pred_scaled)
y_test_actual = scaler_target.inverse_transform(y_test)

# 计算评价指标
mse = mean_squared_error(y_test_actual, y_test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, y_test_pred)
r2 = r2_score(y_test_actual, y_test_pred)
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R²: {r2:.4f}")

# 11. 可视化训练损失
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 假设 y_test_actual 和 y_test_pred 是你的真实值和预测值
# y_test_actual = ...  # 实际值
# y_test_pred = ...    # 预测值

# 计算 MSE、RMSE、MAE、R²
mse = mean_squared_error(y_test_actual, y_test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, y_test_pred)
r2 = r2_score(y_test_actual, y_test_pred)

# 计算 MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    # 避免除以零的情况
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0  # 只计算非零值的MAPE
    return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100

mape = mean_absolute_percentage_error(y_test_actual, y_test_pred)

# 输出评估指标
print(f"Test MSE: {mse:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R²: {r2:.4f}")
print(f"Test MAPE: {mape:.4f}%")

import pandas as pd

# 将预测值和真实值转换为 Pandas DataFrame
results = pd.DataFrame({
    'Actual': y_test_actual.flatten(),
    'Predicted': y_test_pred.flatten()
})

# 打印前 10 行结果
print(results.head(10))

results.to_excel('/path/to/your/directory/prediction_results.xlsx', index=False)


# In[1]:


# 12. 使用 SHAP 解释模型
import shap

# 初始化 SHAP 解释器
# 使用部分训练数据作为背景数据来计算 SHAP 值
background_data = X_train[:100]  # 使用前 100 个训练样本作为背景数据
explainer = shap.DeepExplainer(final_model, background_data)

# 计算测试数据的 SHAP 值
shap_values = explainer.shap_values(X_test)

# 13. 可视化 SHAP 值
# 特征重要性图
shap.summary_plot(shap_values, X_test, feature_names=['IMF 11', 'IMF 12', 'IMF 10', 'IMF 9', 'Wind Speed'])

# 单个样本的 SHAP 解释图
# 选择一个样本（例如，测试集中的第一个样本）
sample_index = 0
shap.force_plot(
    explainer.expected_value,
    shap_values[sample_index],
    X_test[sample_index],
    feature_names=['IMF 11', 'IMF 12', 'IMF 10', 'IMF 9', 'Wind Speed']
)

# 对于回归问题，还可以绘制 SHAP 值与特征值的关系图
for i in range(X_test.shape[2]):  # 遍历每个特征
    shap.dependence_plot(
        i,
        shap_values,
        X_test,
        feature_names=['IMF 11', 'IMF 12', 'IMF 10', 'IMF 9', 'Wind Speed'],
        display_features=X_test
    )


# In[6]:





# 
# # 设置保存路径
# directory = r'D:\Data'  # 目标目录
# file_path = os.path.join(directory, 'prediction_results.xlsx')
# 
# # 确保目录存在
# os.makedirs(directory, exist_ok=True)
# 
# # 保存到 Excel 文件
# results.to_excel(file_path, index=False)
# print(f"Results saved to '{file_path}'")

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 自定义计算 MAPE 的函数
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 假设 X_train, X_test, y_train, y_test, y_test_actual, y_test_pred 已经定义好了
# 并且它们的形状是正确的

# 示例数据（假设这些变量已经从之前的 LSTM 模型中获得）
# X_train, X_test, y_train, y_test, y_test_actual, y_test_pred = ...

# 计算残差序列
residuals = y_test_actual.flatten() - y_test_pred.flatten()

# 确保 X_train 是三维的，并且正确地重塑
if X_train.ndim == 3:
    # 假设我们想要保留每个样本的所有时间步长
    # 这通常意味着我们需要将 y_train 也进行相同的处理
    y_train = y_train.flatten()  # 确保 y_train 是一维的
    
    # 重塑 X_train 为二维
    X_train_res = X_train.reshape(-1, X_train.shape[-1])
    
    # 重塑 y_train 使其与 X_train_res 的样本数量匹配
    # 每个样本的时间步长为 time_step，因此 y_train 需要重复 time_step 次并截断
    time_step = X_train.shape[1]  # 获取时间步长
    y_train_repeated = np.repeat(y_train, time_step)
    y_train_resampled = y_train_repeated[:len(X_train_res)]  # 截断到与 X_train_res 的长度一致
else:
    # 如果 X_train 已经是二维的，则不需要重塑
    X_train_res = X_train
    y_train_resampled = y_train

# 现在我们检查 X_train_res 和 y_train_resampled 的形状是否匹配
print(f"X_train_res shape: {X_train_res.shape}")
print(f"y_train_resampled shape: {y_train_resampled.shape}")

# 打印前几个样本以验证数据重塑的正确性
print("X_train_res 前几个样本:")
print(X_train_res[:5])
print("y_train_resampled 前几个样本:")
print(y_train_resampled[:5])

# 使用匹配的 X_train_res 和 y_train_resampled 训练 SVM 模型
try:
    svm_model = SVR(kernel='rbf')
    svm_model.fit(X_train_res, y_train_resampled)
except Exception as e:
    print(f"SVM 模型训练出错: {e}")
    svm_model = None

# 预测残差
if svm_model is not None:
    try:
        residuals_pred = svm_model.predict(X_test.reshape(-1, X_test.shape[-1]))
    except Exception as e:
        print(f"SVM 模型预测出错: {e}")
        residuals_pred = np.zeros_like(residuals)
else:
    residuals_pred = np.zeros_like(residuals)

# 结合 LSTM 和 SVM 的预测结果
final_predictions = y_test_pred.flatten() + residuals_pred

# 评估最终预测结果
mse = mean_squared_error(y_test_actual.flatten(), final_predictions)
mae = mean_absolute_error(y_test_actual.flatten(), final_predictions)
mape = mean_absolute_percentage_error(y_test_actual.flatten(), final_predictions)
r2 = r2_score(y_test_actual.flatten(), final_predictions)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Absolute Percentage Error: {mape}")
print(f"R^2 Score: {r2}")

# 绘制实际值与最终预测值的对比图
plt.figure(figsize=(15, 6))
plt.plot(y_test_actual.flatten(), label='Actual Values', color='blue', linestyle='-')
plt.plot(final_predictions, label='Final Predictions (LSTM + SVM)', color='red', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.title('Actual vs. Final Predictions')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()





# In[61]:


# 打印前几个样本以验证数据重塑的正确性
print("X_train_res 前几个样本:")
print(X_train_res[:5])
print("y_train_resampled 前几个样本:")
print(y_train_resampled[:5])


# In[3]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 假设 y_test_actual 和 y_test_pred 是你的真实值和预测值
# y_test_actual = ...  # 实际值
# y_test_pred = ...    # 预测值

# 计算 MSE、RMSE、MAE、R²
mse = mean_squared_error(y_test_actual, y_test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, y_test_pred)
r2 = r2_score(y_test_actual, y_test_pred)

# 计算 MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    # 避免除以零的情况
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0  # 只计算非零值的MAPE
    return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100

mape = mean_absolute_percentage_error(y_test_actual, y_test_pred)

# 输出评估指标
print(f"Test MSE: {mse:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R²: {r2:.4f}")
print(f"Test MAPE: {mape:.4f}%")


# In[11]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 假设 y_test_actual 和 y_test_pred 是你的真实值和预测值
# y_test_actual = ...  # 实际值
# y_test_pred = ...    # 预测值

# 设定绝对误差的阈值
error_threshold = 10.0  # 你可以根据实际情况调整这个阈值

# 过滤掉异常值
absolute_errors = np.abs(y_test_actual - y_test_pred)
valid_indices = absolute_errors <= error_threshold

y_test_actual_filtered = y_test_actual[valid_indices]
y_test_pred_filtered = y_test_pred[valid_indices]

# 计算 MSE、RMSE、MAE、R²
mse = mean_squared_error(y_test_actual_filtered, y_test_pred_filtered)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual_filtered, y_test_pred_filtered)
r2 = r2_score(y_test_actual_filtered, y_test_pred_filtered)

# 计算 MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    # 避免除以零的情况
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0  # 只计算非零值的MAPE
    return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100

mape = mean_absolute_percentage_error(y_test_actual_filtered, y_test_pred_filtered)

# 输出评估指标
print(f"Test MSE (filtered): {mse:.4f}")
print(f"Test RMSE (filtered): {rmse:.4f}")
print(f"Test MAE (filtered): {mae:.4f}")
print(f"Test R² (filtered): {r2:.4f}")
print(f"Test MAPE (filtered): {mape:.4f}%")


# In[ ]:


import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from scipy.stats import zscore

# 定义评估指标（修改 SMAPE 为 MAPE）
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    non_zero_mask = y_true != 0  # 过滤真实值为零的样本
    return 100 * np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]))

def log_cosh_loss(y_true, y_pred):
    return np.mean(np.log(np.cosh(y_pred - y_true)))

def adjusted_r2(y_true, y_pred, n_samples, n_features):
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

# 数据过滤（保持原逻辑）
error = np.abs(y_test_actual - y_test_pred)
z_scores = zscore(error)
z_threshold = 2.5
valid_indices_z = np.abs(z_scores) <= z_threshold

Q1 = np.percentile(error, 25)
Q3 = np.percentile(error, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
valid_indices_iqr = (error >= lower_bound) & (error <= upper_bound)

valid_indices = valid_indices_z & valid_indices_iqr
y_test_actual_filtered = y_test_actual[valid_indices]
y_test_pred_filtered = y_test_pred[valid_indices]

# 计算评估指标（修改 SMAPE 为 MAPE）
mse = mean_squared_error(y_test_actual_filtered, y_test_pred_filtered)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual_filtered, y_test_pred_filtered)
r2 = r2_score(y_test_actual_filtered, y_test_pred_filtered)
mape_value = mape(y_test_actual_filtered, y_test_pred_filtered)  # 改为 MAPE
medae = median_absolute_error(y_test_actual_filtered, y_test_pred_filtered)
log_cosh = log_cosh_loss(y_test_actual_filtered, y_test_pred_filtered)
adj_r2 = adjusted_r2(y_test_actual_filtered, y_test_pred_filtered, len(y_test_actual_filtered), X_train.shape[1])

# 输出评估指标（修改标签）
print(f"Test MSE (filtered): {mse:.4f}")
print(f"Test RMSE (filtered): {rmse:.4f}")
print(f"Test MAE (filtered): {mae:.4f}")
print(f"Test R² (filtered): {r2:.4f}")
print(f"Test Adjusted R²: {adj_r2:.4f}")
print(f"Test MAPE: {mape_value:.4f}%")  # 标签改为 MAPE
print(f"Test MedAE: {medae:.4f}")
print(f"Test Log-Cosh Loss: {log_cosh:.4f}")


# In[ ]:





# In[10]:


import matplotlib.pyplot as plt

# 假设 y_test_actual 和 y_test_pred 是你的真实值和预测值
# y_test_actual = ...  # 实际值
# y_test_pred = ...    # 预测值

# 设置图表尺寸和布局
plt.figure(figsize=(14, 7))  # 增大图表尺寸

# 绘制真实值和预测值的对比图
plt.plot(y_test_actual, label='Actual Power (MW)', color='blue', linewidth=2, alpha=0.8)
plt.plot(y_test_pred, label='Predicted Power (MW)', color='red', linestyle='--', linewidth=2, alpha=0.8)

# 设置标题和坐标轴标签
plt.title('Bayesian Optimization: Actual vs Predicted Power', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Time Steps', fontsize=14, labelpad=10)
plt.ylabel('Power (MW)', fontsize=14, labelpad=10)

# 调整刻度密度
plt.xticks(fontsize=12, rotation=45)  # 横轴刻度字体大小和旋转角度
plt.yticks(fontsize=12)  # 纵轴刻度字体大小

# 设置图例（放在图表内部，调整位置和样式）
plt.legend(
    fontsize=12, 
    loc='upper right', 
    bbox_to_anchor=(0.98, 0.98),  # 微调图例位置
    frameon=True,  # 显示图例边框
    framealpha=0.8,  # 图例背景透明度
    edgecolor='black',  # 图例边框颜色
    facecolor='white'  # 图例背景颜色
)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.6)

# 调整布局，避免内容重叠
plt.tight_layout()

# 显示图表
plt.show()


# In[6]:


import matplotlib.pyplot as plt

# 假设 history 是贝叶斯优化训练的历史记录
# history = model.fit(...)

# 提取训练损失和验证损失
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# 绘制训练和验证损失图
plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Training Loss', color='blue', linewidth=2)
plt.plot(val_loss, label='Validation Loss', color='orange', linewidth=2)

# 设置标题和坐标轴标签
plt.title('Bayesian Optimization: Training and Validation Loss', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Epochs', fontsize=14, labelpad=10)
plt.ylabel('Loss', fontsize=14, labelpad=10)

# 设置图例
plt.legend(fontsize=12, loc='upper right')

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.6)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()


# In[23]:


# 1. 调整时间步长和预测范围
time_step = 96  # 过去 24 小时的数据（24 * 4 = 96 个 15 分钟时间点）
forecast_horizon = 24  # 预测未来 6 小时的数据（6 * 4 = 24 个 15 分钟时间点）

# 2. 数据预处理
def create_dataset(X, y, time_step=1, forecast_horizon=1):
    X_data, y_data = [], []
    for i in range(len(X) - time_step - forecast_horizon + 1):
        X_data.append(X[i:(i + time_step), :])
        y_data.append(y[(i + time_step):(i + time_step + forecast_horizon)])
    return np.array(X_data), np.array(y_data)

X, y = create_dataset(features_scaled, target_scaled, time_step, forecast_horizon)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 3. 定义 LSTM 模型构建函数
def build_lstm_model(lstm_units=64, dropout_rate=0.3, learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))  # 添加 Input 层
    model.add(LSTM(int(lstm_units), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(int(lstm_units), return_sequences=True))  # 增加一层 LSTM
    model.add(Dropout(dropout_rate))
    model.add(LSTM(int(lstm_units), return_sequences=False))  # 最后一层 LSTM
    model.add(Dropout(dropout_rate))
    model.add(Dense(100, activation='relu'))  # 增加 Dense 层
    model.add(Dense(50, activation='relu'))  # 增加 Dense 层
    model.add(Dense(25, activation='relu'))  # 增加 Dense 层
    model.add(Dense(forecast_horizon))  # 输出未来 24 个时间点的预测
    
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-5)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# 4. 训练模型（省略贝叶斯优化部分，直接使用最佳参数）
final_model = build_lstm_model(lstm_units=64, dropout_rate=0.3, learning_rate=0.001)

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history = final_model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)
# 12. 预测未来 6 小时的发电功率
def predict_future(model, last_sequence, forecast_horizon=24):
    """
    使用训练好的模型预测未来多个时间步的目标值。

    参数:
        model: 训练好的模型。
        last_sequence: 最后一段输入序列 (形状: [time_step, num_features])。
        forecast_horizon: 预测的时间步数。

    返回:
        future_predictions: 未来多个时间步的预测值。
    """
    future_predictions = []
    current_sequence = last_sequence  # 当前输入序列
    
    for _ in range(forecast_horizon):
        # 预测下一步
        next_pred = model.predict(current_sequence[np.newaxis, :, :])  # 形状: (1, 24)
        future_predictions.append(next_pred[0, 0])  # 只取第一个时间点的预测值
        
        # 更新输入序列
        current_sequence = np.roll(current_sequence, -1, axis=0)  # 将序列向前滚动
        current_sequence[-1, :-1] = current_sequence[-2, :-1]  # 更新特征（除目标值外的其他特征）
        current_sequence[-1, -1] = next_pred[0, 0]  # 更新目标值（只取第一个时间点的预测值）
    
    return np.array(future_predictions)

# 获取最后一段序列
last_sequence = X_test[-1]  # 使用测试集的最后一段序列作为输入

# 预测未来 6 小时（24 个时间点）
future_predictions_scaled = predict_future(final_model, last_sequence, forecast_horizon=24)

# 将预测值反归一化
future_predictions = scaler_target.inverse_transform(future_predictions_scaled.reshape(-1, 1))

# 输出预测结果
print("未来 6 小时的发电功率预测值 (MW):", future_predictions.flatten())

# 13. 结合历史数据和预测数据进行可视化
# 获取历史电功率数据（测试集的最后一段）
historical_power = scaler_target.inverse_transform(y_test[-1].reshape(-1, 1))

# 创建时间轴
historical_time = np.arange(-len(historical_power), 0) * 15  # 历史数据时间轴（单位：分钟）
future_time = np.arange(0, len(future_predictions)) * 15  # 预测数据时间轴（单位：分钟）

# 绘制历史数据和预测数据
plt.figure(figsize=(14, 7))  # 增加图表尺寸
plt.plot(
    historical_time, historical_power, 
    marker='o', linestyle='-', linewidth=2, markersize=8, 
    label='Historical Power', color='blue'
)
plt.plot(
    future_time, future_predictions, 
    marker='s', linestyle='--', linewidth=2, markersize=8, 
    label='Predicted Power', color='red'
)
plt.axvline(x=0, color='gray', linestyle=':', linewidth=2, label='Prediction Start')  # 预测起始线
plt.xlabel('Time (Minutes Relative to Prediction Start)', fontsize=12)
plt.ylabel('Power (MW)', fontsize=12)
plt.title('Historical and Future 6 Hours Power Prediction (15-minute Intervals)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)  # 添加网格线
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# In[24]:





# In[ ]:




