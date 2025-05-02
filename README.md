# Wind Power Forecast Webapp

## 项目简介

``` document
wind_power_forecast_webapp/
│
├── app/                              # 应用核心目录
│   ├── __init__.py                   # Flask应用初始化
│   ├── routes.py                     # 所有路由配置
│   ├── forms.py                      # 表单类定义（上传/参数配置）
│   ├── models/                       # 机器学习模型模块
│   │   ├── m_configure.json          # 模型配置参数
│   │   ├── ceemdan_decomposer.py     # CEEMDAN信号分解实现
│   │   ├── bayesian_lstm.py          # 贝叶斯优化LSTM模型
│   │   ├── residual_rf.py            # 随机森林残差修正模块
│   │   └── model_utils.py            # 模型加载/保存工具
│   │
│   ├── utils/                        # 数据处理工具
│   │   ├── p_configure.json          # 数据预处理配置参数
│   │   ├── data_cleaner.py           # DPGMM异常检测与数据清洗
│   │   ├── DPGMM.py                  # DPGMM核心算法
│   │   ├── feature_engineer.py       # 特征工程与IMF选择
│   │   ├── scaler_manager.py         # 归一化处理器管理
│   │   └── shap_analyzer.py          # SHAP解释性分析模块
│   │
│   ├── templates/                    # 前端模板
│   │   ├── base.html                 # 基础模板
│   │   ├── index.html                # 仪表盘主页
│   │   ├── upload.html               # 数据上传界面
│   │   ├── training.html             # 模型训练监控界面
│   │   ├── results.html              # 多维结果展示界面
│   │   ├── predict.html              # 预测结果展示界面
│   │   ├── preprocessing.html        # 数据预处理界面
│   │   └── report.html               # 综合效益评估报告
│   │
│   ├── static/                       # 静态资源
│   │   ├── css/                      # 样式表
│   │   │   ├── dashboard.css         # 主界面样式
│   │   │   └── charts.css            # 可视化专用样式
│   │   │
│   │   ├── js/                       # JavaScript脚本
│   │   │   ├── realtime_charts.js    # 实时训练监控
│   │   │   ├── data_visualization.js # 交互式可视化
│   │   │   └── api_handler.js        # 前后端交互
│   │   │
│   │   └── assets/                   # 静态资源
│   │       ├── imgs/                 # 图片资源
│   │       └── model_diagrams/       # 技术架构图
│   │
│   ├── data/                         # 数据管理
│   │   ├── raw/                      # 原始数据存储
│   │   ├── cleaned/                  # 清洗后数据存储
│   │   ├── processed/                # 处理后数据存储
│   │   └── predictions/              # 预测结果存储
│   │
│   └── tasks/                        # 异步任务模块
│       ├── celery_config.py          # Celery配置
│       ├── training_tasks.py         # 模型训练任务
│       └── prediction_tasks.py       # 批量预测任务
│
├── config/                           # 系统配置
│   ├── settings.py                   # 全局配置参数
│   ├── constants.py                  # 常量定义
│   └── logging.conf                  # 日志配置
│
├── tests/                            # 测试模块
│   ├── test_data_processing.py       # 数据处理测试
│   ├── test_model_performance.py     # 模型性能测试
│   └── test_api_endpoints.py         # API接口测试
│
├── requirements.txt                  # Python依赖库
├── Dockerfile                        # 容器化部署配置
├── docker-compose.yml                # 服务编排文件
├── run.py                            # 启动脚本
└── README.md                         # 项目说明文档
```

### 核心文件说明

1. **模型核心模块（models/）**
   - `bayesian_lstm.py`：实现贝叶斯优化LSTM的超参数搜索与早停机制
   - `residual_rf.py`：包含随机森林残差修正的动态窗口调整算法
   - `ceemdan_decomposer.py`：CEEMDAN信号分解的并行化实现

2. **数据处理模块（utils/）**
   - `data_cleaner.py`：实现基于DPGMM的动态异常检测流程
   - `feature_engineer.py`：包含IMF分量选择与风速特征融合算法
   - `shap_analyzer.py`：提供模型可解释性分析的SHAP值计算

3. **可视化系统（static/js/）**
   - `realtime_charts.js`：使用ECharts实现：
   - 训练损失曲线实时更新
   - 预测结果对比动态渲染
   - SHAP特征重要性瀑布图

4. **异步任务（tasks/）**
   - `training_tasks.py`：包含：
   - 贝叶斯优化任务的分布式计算
   - 模型训练进度实时回调
   - 大文件预测的内存优化处理

5. **系统特色功能**
   - 多风电场集群预测模式切换
   - 预测结果与综合效益（碳减排量）联动计算
   - 支持.xlsx/.csv/.h5多种数据格式上传
   - 模型配置参数的可视化调试界面

### 技术栈建议

- **前端**：Vue3 + ECharts + Element Plus
- **后端**：Flask + Celery + Redis
- **数据处理**：Pandas + NumPy + Dask（大数据处理）
- **机器学习**：TensorFlow/Keras + scikit-learn + SHAP
- **部署**：Docker + Nginx + Gunicorn
