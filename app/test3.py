from flask import Flask, request, render_template, jsonify, redirect, url_for
import os
from werkzeug.utils import secure_filename
from data_manager import * 
import json



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/raw'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/clean_data', methods=['GET'])
def clean_data():
    """清理数据"""
    return render_template('clean_data.html')


@app.route('/get_config')
def get_config():
    """获取当前配置（修复空值问题）"""
    default_config = {
        "current_file": "",
        "file_options": [],
        "anomaly_handling": {
            "method": "delete",
            "interpolate_method": "linear"
        }
    }
    
    try:
        with open('utils/p_configure.json') as f:
            config = json.load(f)
            print('参数配置：', config)
    except FileNotFoundError:
        config = default_config
    
    # 动态更新文件列表
    config['file_options'] = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                             if f.endswith('.csv')]
    return jsonify(config)


@app.route('/update_config', methods=['POST'])
def update_config():
    """更新配置文件（智能合并）"""
    try:
        # 读取当前完整配置
        with open('utils/p_configure.json', 'r') as f:
            current_config = json.load(f)
    except FileNotFoundError:
        current_config = create_default_config()  # 创建默认配置

    # 获取前端提交的局部配置
    partial_update = request.json
    
    # 深度合并配置
    def deep_merge(base, update):
        """递归合并嵌套字典"""
        for key, value in update.items():
            if isinstance(value, dict):
                base[key] = deep_merge(base.get(key, {}), value)
            else:
                base[key] = value
        return base
    
    merged_config = deep_merge(current_config, partial_update)

    # 写回文件
    with open('utils/p_configure.json', 'w') as f:
        json.dump(merged_config, f, indent=4, ensure_ascii=False)
    
    return jsonify({'status': 'success'})



@app.route('/process_data', methods=['POST'])
def process_data():
    """执行数据处理"""
    try:
        # 获取配置
        with open('utils/p_configure.json') as f:
            config = json.load(f)
        
        # 调用数据清洗模块
        raw_path = os.path.join(app.config['UPLOAD_FOLDER'], config['current_file'])
        # processor = DataProcessor(config)
        # result = processor.full_pipeline(raw_path)
        render_template('errors.html', error_message={
            'status': 'processing_error',
           'message': '数据处理失败，请检查配置或联系管理员'
        })
        
        return 
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


def create_default_config():
    """生成默认配置结构"""
    return {
            "current_file": "wind_data_2024.csv",
            "file_options": ["file1.csv", "file2.csv"],
            "anomaly_handling": {
                "method": "delete",
                "interpolate_method": "linear",
                "max_anomaly_duration": 3,
                "threshold": 2.5
            },
            "data_format":{
                "missing_value_strategy": "interpolate",  
                "max_missing_ratio": 0.3, 
                "anomaly_handling": {
                    "method": "knn",
                    "interpolate_method": "linear"
                }
            },
            "feature_settings": {
                "window_size": 3,
                "ceemdan_levels": 5,
                "enable_meteorology": True
            },
            "system": {
                "last_modified": "2024-03-20 14:30:00",
                "version": "1.2.0"
            }
            }


if __name__=='__main__':
    app.run(debug=True)