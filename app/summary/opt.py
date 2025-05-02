import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ====================
# 1. 数据准备模块
# ====================
def load_forecast_data():
    """模拟加载传统模型和优化模型的预测数据"""
    # 时间戳（15分钟间隔）
    time_index = pd.date_range(start="2025-03-01 00:00", periods=96, freq="15T")
    
    # 传统模型预测（示例数据）
    traditional_forecast = np.abs(np.random.normal(loc=50, scale=15, size=96))  # 均值50MW，标准差15
    traditional_forecast = np.clip(traditional_forecast, 0, 100)  # 限制在0-100MW
    
    # 优化模型预测（示例数据）
    optimized_forecast = traditional_forecast + np.random.normal(0, 3, 96)  # 添加噪声模拟优化效果
    optimized_forecast = np.clip(optimized_forecast, 0, 100)
    
    # 实际功率（模拟真实值）
    actual_power = traditional_forecast + np.random.normal(0, 8, 96)
    actual_power = np.clip(actual_power, 0, 100)
    
    return pd.DataFrame({
        'time': time_index,
        'traditional': traditional_forecast,
        'optimized': optimized_forecast,
        'actual': actual_power
    }).set_index('time')

# ====================
# 2. 调度参数设置
# ====================
class SchedulingParams:
    def __init__(self):
        # 火电机组参数（示例：3台机组）
        self.thermal_units = {
            'cost_coeff': [0.05, 0.04, 0.06],  # 成本系数 $/MW²h
            'p_max': [80, 100, 60],            # 最大出力(MW)
            'ramp_rate': [20, 25, 15]          # 爬坡速率(MW/15min)
        }
        self.reserve_cost = 10                  # 备用成本 $/MW
        self.co2_factor = 0.85                 # 煤电CO2排放因子 t/MWh

# ====================
# 3. 调度优化计算
# ====================
def dispatch_optimization(forecast, params):
    """基于预测值的调度优化"""
    # 目标函数：最小化总成本（发电成本 + 备用成本）
    def cost_function(p):
        generation = p[:3]  # 前三变量为机组出力
        reserve = p[3]       # 备用容量
        # 发电成本（二次函数）
        gen_cost = sum(c * (g**2) for c, g in zip(params.thermal_units['cost_coeff'], generation))
        # 备用成本
        reserve_cost = reserve * params.reserve_cost
        return gen_cost + reserve_cost
    
    # 约束条件
    constraints = [
        # 能量平衡约束
        {'type': 'eq', 'fun': lambda p: sum(p[:3]) + forecast - actual_power},  # 假设已知实际值
        # 机组出力约束
        {'type': 'ineq', 'fun': lambda p: p[0]},  # 出力下限
        {'type': 'ineq', 'fun': lambda p: params.thermal_units['p_max'][0] - p[0]},
        # ... 其他机组约束（需扩展）
    ]
    
    # 初始猜测（平分预测值）
    initial_guess = [forecast/3, forecast/3, forecast/3, 0]
    
    # 求解优化问题
    result = minimize(cost_function, initial_guess, constraints=constraints)
    return result.x

# ====================
# 4. 调度结果评估
# ====================
def evaluate_scheduling(df, params):
    """评估两种预测模型的调度效果"""
    results = {}
    for model in ['traditional', 'optimized']:
        # 计算调度结果
        dispatch_results = [dispatch_optimization(p, params) for p in df[model]]
        
        # 提取关键指标
        total_gen = np.sum([sum(res[:3]) for res in dispatch_results])
        reserve_used = np.mean([res[3] for res in dispatch_results])
        curtailment = np.sum(np.maximum(df[model] - df['actual'], 0))  # 弃风量
        
        # 存储结果
        results[model] = {
            'total_cost': sum(res[-1] for res in dispatch_results),
            'co2_emission': (total_gen * params.co2_factor) / 1e3,  # 吨CO2
            'reserve_used': reserve_used,
            'curtailment_rate': curtailment / df[model].sum()
        }
    return results

# ====================
# 5. 可视化模块
# ====================
def visualize_results(df, results):
    """可视化预测误差与调度效果"""
    plt.figure(figsize=(15, 10))
    
    # 预测对比曲线
    plt.subplot(2, 2, 1)
    df[['traditional', 'optimized', 'actual']].plot(title="Wind Power Forecast Comparison")
    plt.ylabel("Power (MW)")
    
    # 调度成本对比
    plt.subplot(2, 2, 2)
    plt.bar(results.keys(), [v['total_cost'] for v in results.values()])
    plt.title("Total Dispatch Cost Comparison")
    plt.ylabel("Cost ($)")
    
    # 弃风率对比
    plt.subplot(2, 2, 3)
    plt.bar(results.keys(), [v['curtailment_rate']*100 for v in results.values()])
    plt.title("Curtailment Rate Comparison")
    plt.ylabel("Curtailment (%)")
    
    # CO2排放对比
    plt.subplot(2, 2, 4)
    plt.bar(results.keys(), [v['co2_emission'] for v in results.values()])
    plt.title("CO2 Emissions Comparison")
    plt.ylabel("CO2 (tons)")
    
    plt.tight_layout()
    plt.show()

# ====================
# 主程序
# ====================
if __name__ == "__main__":
    # 加载数据
    df = load_forecast_data()
    
    # 参数初始化
    params = SchedulingParams()
    
    # 评估调度效果
    results = evaluate_scheduling(df, params)
    
    # 打印关键指标
    print("=== 调度效果评估 ===")
    for model in ['traditional', 'optimized']:
        print(f"\n** {model}模型 **")
        print(f"总成本: {results[model]['total_cost']:.2f} $")
        print(f"弃风率: {results[model]['curtailment_rate']*100:.2f}%")
        print(f"CO2排放: {results[model]['co2_emission']:.2f} 吨")
    
    # 可视化结果
    visualize_results(df, results)