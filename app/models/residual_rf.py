import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from bayes_opt import BayesianOptimization

def run_model_evaluation():
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

    time_step = 96
    X, y = create_dataset(features_scaled, target_scaled, time_step)

    # 划分训练集和测试集
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 4. 定义 LSTM 模型构建函数
    def build_lstm_model(lstm_units=64, dropout_rate=0.3, learning_rate=0.001):
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(int(lstm_units), return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(int(lstm_units), return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(int(lstm_units), return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1, activation='relu'))
        
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-5)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    # 5. 定义贝叶斯优化的目标函数
    def optimize_lstm(lstm_units, dropout_rate, learning_rate):
        lstm_units = int(lstm_units)
        dropout_rate = max(min(dropout_rate, 0.5), 0.1)
        learning_rate = max(min(learning_rate, 0.01), 0.0001)

        model = build_lstm_model(lstm_units, dropout_rate, learning_rate)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, lr_scheduler],
            verbose=0
        )
        
        return -history.history['val_loss'][-1]

    # 6. 设置贝叶斯优化的参数范围
    pbounds = {
        'lstm_units': (50, 128),
        'dropout_rate': (0.2, 0.5),
        'learning_rate': (0.0001, 0.01)
    }

    # 7. 运行贝叶斯优化
    optimizer = BayesianOptimization(
        f=optimize_lstm,
        pbounds=pbounds,
        random_state=42
    )

    optimizer.maximize(init_points=5, n_iter=20)

    # 8. 获取最佳超参数
    best_params = optimizer.max['params']
    print("Best Parameters:", best_params)

    # 9. 使用最佳超参数训练最终模型
    final_model = build_lstm_model(
        lstm_units=int(best_params['lstm_units']),
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate']
    )

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

    # 10. 在测试集上进行预测
    y_test_pred_scaled = final_model.predict(X_test)
    y_test_pred = scaler_target.inverse_transform(y_test_pred_scaled)
    y_test_actual = scaler_target.inverse_transform(y_test)

    # 计算评价指标
    mse = mean_squared_error(y_test_actual, y_test_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, y_test_pred)
    r2 = r2_score(y_test_actual, y_test_pred)

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        non_zero_indices = y_true != 0
        return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100

    mape = mean_absolute_percentage_error(y_test_actual, y_test_pred)

    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")
    print(f"Test MAPE: {mape:.4f}%")

    # 11. 可视化训练损失
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # 12. 预测未来 6 小时（24 个时间点）
    forecast_horizon = 24

    def predict_future(model, last_sequence, forecast_horizon=24):
        future_predictions = []
        current_sequence = last_sequence  # 当前输入序列
        
        for _ in range(forecast_horizon):
            next_pred = model.predict(current_sequence[np.newaxis, :, :])  
            future_predictions.append(next_pred[0, 0])  
            
            current_sequence = np.roll(current_sequence, -1, axis=0)  
            current_sequence[-1, :-1] = current_sequence[-2, :-1]  
            current_sequence[-1, -1] = next_pred[0, 0] 
        
        return np.array(future_predictions)

    last_sequence = X_test[-1]
    future_predictions_scaled = predict_future(final_model, last_sequence, forecast_horizon=24)
    future_predictions = scaler_target.inverse_transform(future_predictions_scaled.reshape(-1, 1))

    print("未来 6 小时的发电功率预测值 (MW):", future_predictions.flatten())

    # 13. 结合历史数据和预测数据进行可视化
    historical_power = scaler_target.inverse_transform(y_test[-1].reshape(-1, 1))
    historical_time = np.arange(-len(historical_power), 0) * 15 
    future_time = np.arange(0, len(future_predictions)) * 15  

    plt.figure(figsize=(14, 7))
    plt.plot(historical_time, historical_power, marker='o', linestyle='-', linewidth=2, markersize=8, label='Historical Power', color='blue')
    plt.plot(future_time, future_predictions, marker='s', linestyle='--', linewidth=2, markersize=8, label='Predicted Power', color='red')
    plt.axvline(x=0, color='gray', linestyle=':', linewidth=2, label='Prediction Start')
    plt.xlabel('Time (Minutes Relative to Prediction Start)', fontsize=12)
    plt.ylabel('Power (MW)', fontsize=12)
    plt.title('Historical and Future 6 Hours Power Prediction (15-minute Intervals)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, None)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    # 14. 计算主模型残差
    print("\n正在计算主模型残差...")

    y_train_pred_scaled = final_model.predict(X_train)
    y_train_pred = scaler_target.inverse_transform(y_train_pred_scaled)
    y_train_actual = scaler_target.inverse_transform(y_train)
    residuals_train = y_train_actual - y_train_pred

    y_test_pred_scaled = final_model.predict(X_test)
    y_test_pred = scaler_target.inverse_transform(y_test_pred_scaled)
    y_test_actual = scaler_target.inverse_transform(y_test)
    residuals_test = y_test_actual - y_test_pred

    scaler_residual = MinMaxScaler(feature_range=(0, 1))
    residuals_train_scaled = scaler_residual.fit_transform(residuals_train)
    residuals_test_scaled = scaler_residual.transform(residuals_test)

    # 15. 训练随机森林回归模型
    print("\n训练随机森林残差模型...")

    residual_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=4,
        random_state=42
    )

    residual_model.fit(X_train.reshape(X_train.shape[0], -1), residuals_train_scaled)

    # 16. 预测测试集残差
    print("\n预测测试集残差...")
    residuals_pred_test_scaled = residual_model.predict(X_test.reshape(X_test.shape[0], -1))
    residuals_pred_test = scaler_residual.inverse_transform(residuals_pred_test_scaled.reshape(-1, 1))

    # 17. 计算联合预测值
    print("\n计算联合预测值...")
    final_test_pred = y_test_pred + residuals_pred_test

    # 18. 评估联合预测效果
    print("\n修正后测试集指标:")
    mse_final = mean_squared_error(y_test_actual, final_test_pred)
    rmse_final = np.sqrt(mse_final)
    mae_final = mean_absolute_error(y_test_actual, final_test_pred)
    r2_final = r2_score(y_test_actual, final_test_pred)
    mape_final = mean_absolute_percentage_error(y_test_actual, final_test_pred)

    print(f"Test RMSE: {rmse_final:.4f}")
    print(f"Test MAE: {mae_final:.4f}")
    print(f"Test R²: {r2_final:.4f}")
    print(f"Test MAPE: {mape_final:.4f}%")

    # 19. 可视化主模型与联合模型对比
    plt.figure(figsize=(14, 6))
    plt.plot(y_test_actual[-100:], label='Actual Power', color='#2c7bb6', linewidth=2)
    plt.plot(final_test_pred[-100:], label='Combined Prediction', color='#d7191c', linestyle='--', linewidth=2)
    plt.plot(y_test_pred[-100:], label='LSTM Prediction', color='#fdae61', linestyle=':', linewidth=2)
    plt.axvline(x=len(y_test_actual)-24, color='gray', linestyle='--', label='Prediction Start')
    plt.title('Model Comparison: LSTM vs Combined Model', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Power (MW)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()

    time_steps_to_plot = 100
    plt.figure(figsize=(14, 6))
    plt.plot(y_test_actual[:time_steps_to_plot], label='Actual Power', color='#2c7bb6', linewidth=2)
    plt.plot(final_test_pred[:time_steps_to_plot], label='Combined Prediction', color='#d7191c', linestyle='--', linewidth=2)
    plt.plot(y_test_pred[:time_steps_to_plot], label='LSTM Prediction', color='#fdae61', linestyle=':', linewidth=2)
    prediction_start_index = time_steps_to_plot - 24
    plt.axvline(x=prediction_start_index, color='gray', linestyle='--', label='Prediction Start')
    plt.title('Model Comparison: LSTM vs Combined Model', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Power (MW)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()

# 调用函数
run_model_evaluation()