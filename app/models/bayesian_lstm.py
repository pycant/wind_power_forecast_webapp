import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

def run_lstm_model(imf_path, data_path):
    # 1. 加载数据
    imfs_file_path = imf_path
    data_file_path = data_path

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
    def create_dataset(X, y, time_step=10):
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
        model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(int(lstm_units), return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(int(lstm_units), return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(Dense(25))
        model.add(Dense(1))
        
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-5)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    # 5. 定义贝叶斯优化的目标函数
    def optimize_lstm(lstm_units, dropout_rate, learning_rate):
        lstm_units = int(lstm_units)
        dropout_rate = max(min(dropout_rate, 0.5), 0.1)
        learning_rate = max(min(learning_rate, 0.01), 0.0001)

        model = build_lstm_model(lstm_units, dropout_rate, learning_rate)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        return -history.history['val_loss'][-1]

    # 6. 设置贝叶斯优化的参数范围
    pbounds = {
        'lstm_units': (30, 100),
        'dropout_rate': (0.1, 0.5),
        'learning_rate': (0.0001, 0.01)
    }

    # 7. 运行贝叶斯优化
    optimizer = BayesianOptimization(
        f=optimize_lstm,
        pbounds=pbounds,
        random_state=42
    )

    optimizer.maximize(init_points=5, n_iter=15)

    # 8. 获取最佳超参数
    best_params = optimizer.max['params']
    best_params['lstm_units'] = int(best_params['lstm_units'])
    print("Best Parameters:", best_params)

    # 9. 使用最佳超参数训练最终模型
    final_model = build_lstm_model(
        lstm_units=best_params['lstm_units'],
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = final_model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # 10. 在测试集上进行预测
    y_test_pred_scaled = final_model.predict(X_test)
    y_test_pred = scaler_target.inverse_transform(y_test_pred_scaled)
    y_test_actual = scaler_target.inverse_transform(y_test)

    # 计算评价指标
    mse = mean_squared_error(y_test_actual, y_test_pred)
    rmse = np.sqrt(mse)
    print(f"Test RMSE: {rmse:.4f}")

    # 11. 可视化训练损失
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # 12. 可视化测试集的真实值和预测值
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label='Actual Power (MW)', color='blue')
    plt.plot(y_test_pred, label='Predicted Power (MW)', color='red', linestyle='--')
    plt.xlabel('Time Steps')
    plt.ylabel('Power (MW)')
    plt.title('Actual vs Predicted Power on Test Set')
    plt.legend()
    plt.show()

