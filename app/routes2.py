from flask import Flask, render_template, request, jsonify, redirect, url_for,render_template
import os
import uuid
from werkzeug.utils import secure_filename
import tools.data_manager as dm
from tools.data_manager import *
from utils.data_cleaner import *
import unicodedata
import re
import plotly.graph_objects as go
import statsmodels.api as sm
import json
from models.model_utils import *
from flask_cors import CORS 
from functools import wraps
import time

# from models.bayesian_lstm import BayesianLSTMPredictor
# from models.models import * 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/raw'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'csv'}
print( os.path.join(app.config['UPLOAD_FOLDER']))

# 初始化上传目录
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
def is_file_duplicate(target_file, target_dir=os.path):
    # 提取目标文件的文件名
    target_filename = os.path.basename(target_file)
    # 获取指定目录下的所有文件名
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file == target_filename:
                return True
    return False


def validate_content_type(content_types):
    """内容类型验证装饰器"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if request.method in ['POST', 'PUT', 'PATCH']:
                if request.content_type not in content_types:
                    return jsonify({
                        "status": "error",
                        "code": "INVALID_CONTENT_TYPE",
                        "message": f"Unsupported media type: {request.content_type}"
                    }), 415
            return f(*args, **kwargs)
        return wrapper
    return decorator

def handle_errors(f):
    """全局错误处理装饰器"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            app.logger.error(f"请求处理失败: {str(e)}", exc_info=True)
            return jsonify({
                "status": "error",
                "code": "INTERNAL_ERROR",
                "message": "服务器内部错误"
            }), 500
    return wrapper



@app.route('/')
def index():
    print("访问首页")
    if request.method == 'POST':
        print("接收到POST请求", request.form)
    return render_template('base.html')




@app.route('/upload', methods=['POST'])
def upload_file():
    print("接收到文件上传请求", request.files, request.form,sep='\n')
    # 获取前端交互元素信息
    button_clicked = request.form.get('button_id')
    user_agent = request.headers.get('User-Agent')
    
    # 实时输出用户操作日志
    print(f" Button: {button_clicked}, Client: {user_agent}")

    if 'file' not in request.files:
        return render_template('errors.html',error_message={
            'status': 'validation_error',
            'errors': [ '未选择文件']
        })
        
    file = request.files['file']
    if file.filename == '':
        return render_template('errors.html',error_message={
            'status': 'validation_error',
            'errors': ['未选择文件，请重新上传']
        })

    try:
        print(f"开始上传文件：{file.filename}")
        # 前端验证反馈
        if not allowed_file(file.filename):
            return render_template('errors.html',error_message={
                'status': 'validation_error',
                'errors': ['文件格式不支持，请上传CSV文件']
            })
        print(f"文件格式验证通过：{file.filename}")
        
        # # 文件重名检查
        filename = secure_filename(file.filename)
        # print('local filename:',os.path.join(app.config['UPLOAD_FOLDER'],filename))
        # print('文件名:',filename)
        # print(f"原始文件名: {file.filename} → 处理后: {filename}")
        # print(f"完整路径: {filepath}")
        # if os.path.exists(filepath):
            
        #     return render_template('errors.html',error_message={
        #         'status': 'validation_error',
        #         'errors': ['文件名已存在，请修改后重新上传']
        #     })
            
        print(f"文件名验证通过：{file.filename}")
        # 保存文件
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # 记录成功日志
        print(f"文件上传成功：{filename}，大小：{os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], filename))}字节")
        # time.sleep(1000)
        
        return render_template('base.html', success_message={
            'status': 'upload_success',
            'filename': filename,
        })
    except Exception as e:
        # 错误处理及实时反馈
        error_message = f"文件上传失败：{str(e)}"
        print(error_message)
        return render_template('errors.html',error_message={
            'status': 'upload_error',
            'message': error_message
        })


@app.route('/file_manager', methods=['GET','POST'])
def file_manager():
    if request.method == 'GET':
        print("渲染文件管理页面")
        file_stats=get_file_stats()
        return render_template('data_file_manager.html',
                                file_stats=file_stats)        
    if request.method == 'POST':
        print("接收到文件管理请求", request.form, request.files,sep='\n')
        # 获取前端交互元素信息
        button_clicked = request.form.get('button_id')
        user_agent = request.headers.get('User-Agent')
        print(f" Button: {button_clicked}, Client: {user_agent}")
        if button_clicked == 'open_manager':
            print("打开文件管理")
            file_stats=get_file_stats()
            return render_template('data_file_manager.html',
                                    file_stats=file_stats)
    return render_template('errors.html',error_message={
        'status': 'unknown_action',
        'message': f'未知操作：{button_clicked,request.form}'
    })
                


@app.route('/api/files',methods=['GET','POST'])
def delete_data_file():
    """删除文件"""
    try:
        # 获取前端交互元素信息
        relative_path = request.json.get('path')
        print(f"接收到文件删除请求：{relative_path}")
        # 调用文件删除函数
        delete_file(relative_path)
        # 记录成功日志
        print(f"文件删除成功：{relative_path}")
        # 重新加载文件管理页面
        return jsonify({ 'success':True, 'path': relative_path})
    except Exception as e:
        # 错误处理及实时反馈
        error_message = f"文件删除失败：{str(e)}"
        print(error_message)
    


        

@app.route('/data_preprocessing_submit', methods=['post'])
def clean_data():
    """清理数据"""
    print(request.method, request.form)
    try:
        # 读取当前完整配置
        with open('utils/p_configure.json', 'r') as f:
            current_config = json.load(f)
    except FileNotFoundError:
        current_config = create_default_config()

    return render_template('clean_data.html',msg=None)




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
    """智能合并配置（保留原有配置项）"""
    try:
        # 读取当前完整配置
        with open('utils/p_configure.json', 'r') as f:
            current_config = json.load(f)
    except FileNotFoundError:
        current_config = create_default_config()

    def smart_merge(base, update):
        """智能合并配置"""
        for key, value in update.items():
            # 如果是字典类型则递归合并
            if isinstance(value, dict) and key in base:
                smart_merge(base[key], value)
            else:
                # 只更新存在的配置项
                if key in base:
                    base[key] = value
        return base

    # 执行智能合并
    merged_config = smart_merge(current_config, request.json)

    # 写回文件
    with open('utils/p_configure.json', 'w') as f:
        json.dump(merged_config, f, indent=4, ensure_ascii=False)
    
    return jsonify({'status': 'success', 'merged_keys': list(request.json.keys())})



@app.route('/process_data', methods=['POST'])
def process_data():
    try:
        with open('utils/p_configure.json') as f:
            config = json.load(f)
            
        cleaner = DataCleaner()
        cleaned_data = cleaner.format_data()  # 确保返回处理后的数据
        save_path, report = cleaner.save_processed_data()
        p_result_path.append(save_path)
        print("detected_cols",cleaner.message)
        # 构造响应数据
        return jsonify({
            'status': 'success',
            'preview': cleaned_data.head(10).to_html(classes='table table-striped', index=False),
            'message': cleaner.message
        })
        
    except Exception as e:
        # 返回结构化的错误信息
        print(f"处理失败: {str(e)}")
        error_message = {
            'errors': [str(e)],
            'warnings': [],
            'status': []
        }
        return jsonify({
            'status': 'error',
            'message': {
                'errors': [str(e)],
                'warnings': [],
                'status': []
            }
        })
        
    except Exception as e:
        # print(f"处理失败: {str(e)}")
        app.logger.error(f"处理失败: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': {
                'errors': [str(e)],
                'warnings': [],
                'status': []
            }
        })


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
                    "method": "drop",
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


@app.route('/back_to_home', methods=['POST'])
def save_processed_data():
    def plot_data(path):
        # 读取数据
        print("图片绘制中...")
        df = pd.read_csv(path)  # 先普通读取
        df=df.iloc[:3000,:]  
        date_column = df.columns[0]  # 获取第1列名称（位置索引0）
        # df['Time(year-month-day h:m:s)'] = pd.to_datetime(df[date_column])
        time = df.iloc[:, 0]
        power = df.iloc[:, -1]
        
        # LOWESS拟合
        x = np.arange(len(time))
        lowess = sm.nonparametric.lowess(power, x, frac=0.05)
        fitted_power = lowess[:, 1]

        # 创建动态折线图
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time,
            y=power,
            mode='lines',
            name='实际功率',
            line=dict(color='#1f77b4')
        ))
        fig.add_trace(go.Scatter(
            x=time,
            y=fitted_power,
            mode='lines',
            name='LOWESS拟合',
            line=dict(color='#ff7f0e', dash='dot')
        ))

        # 增强图表交互功能
        fig.update_layout(
            title='风电场发电功率趋势分析',
            xaxis_title='时间',
            yaxis_title='功率 (MW)',
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=40, r=20, t=40, b=20)
        )
        
        # 添加异常点标注（示例数据）
        anomalies = power[(power - fitted_power).abs() > 2*power.std()]
        fig.add_trace(go.Scatter(
            x=time[anomalies.index],
            y=anomalies,
            mode='markers',
            name='异常点',
            marker=dict(color='red', size=6)
        ))
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    try:
        # 生成图表HTML
        fig_html = plot_data(p_result_path[-1])
        
        # 获取文件统计信息
        file_stats = get_file_stats()
        
        return render_template('base.html', 
                             chart_html=fig_html,
                             file_stats=file_stats,
                             processed_file=os.path.basename(p_result_path[-1]))
        print("图表生成成功")
    except Exception as e:
        app.logger.error(f"图表生成失败: {str(e)}")
        return render_template('base.html', 
                             error_message="图表生成失败，请检查数据格式")



# 其他模块的交互端点
@app.route('/api/interaction', methods=['POST'])
def handle_interaction():
    # 获取所有交互元素信息
    button_id = request.json.get('button_id')
    input_values = request.json.get('inputs', {})
    
    # 实时记录交互日志
    print(f"用户交互事件 - 按钮: {button_id}, 输入值: {input_values}")
    
    # 根据不同按钮ID处理业务逻辑
    if button_id == 'preprocess_apply':
        # 执行预处理逻辑
        return jsonify({
            'status': 'preprocessing',
            'message': '数据预处理进行中...',
            'progress': 0
        })
        
    elif button_id == 'model_select':
        # 处理模型选择
        return jsonify({
            'status': 'model_selected',
            'model_name': 'CEEMDAN-LSTM',
            'parameters': {'layers': 3}
        })
    
    return jsonify({'status': 'unknown_action'}), 400

MODEL_CONFIG_PATH = 'models/m_configure.json'

@app.route('/model_selection_submit', methods=['POST'])
@handle_errors
def model_selection_submit():
    # 获取选择的模型ID
    selected_id = request.form.get('selected_model_id')
    print(f"接收到模型选择请求，选择ID: {selected_id}")
    
    # 建立模型ID与名称的映射关系
    MODEL_MAPPING = {
        'select-model-1': '基础LSTM',
        'select-model-2': 'CEEMDAN-LSTM'
    }
    
    # 验证模型ID有效性
    if selected_id not in MODEL_MAPPING:
        return jsonify({
            "status": "error",
            "code": "INVALID_MODEL_ID",
            "message": "无效的模型选择标识"
        }), 400
    
    model_name = MODEL_MAPPING[selected_id]
    
    try:
        # 读取现有配置
        if os.path.exists(MODEL_CONFIG_PATH):
            with open(MODEL_CONFIG_PATH, 'r') as f:
                config = json.load(f)
        else:
            config = {
                "model_name": "Bayesian_LSTM",
                "predict_length": 24,
                "predict_type": {
                    "type": "multi",
                    "steps": 24,
                    "confidence": True,
                    "confidence_level": 95
                }
            }
        
        # 更新模型名称
        config['model_name'] = model_name
        config['last_modified'] = datetime.now().isoformat()
        print(f"更新模型配置：{config}")
        # 写入配置文件
        with open(MODEL_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
            
        print(f"模型配置更新成功，当前模型：{model_name}")
        
        return jsonify({
            "status": "success",
            "message": f"模型配置已更新为：{model_name}",
            "model_type": model_name,
            "config": config
        })
        
    except Exception as e:
        app.logger.error(f"模型配置更新失败：{str(e)}")
        return jsonify({
            "status": "error",
            "code": "CONFIG_UPDATE_FAILED",
            "message": f"配置保存失败：{str(e)}"
        }), 500
        
        

@app.route('/predict_submit', methods=['POST'])
def predict_submit():
    try:
        # 从请求获取参数
        predict_type = request.form.get('predict-type')
        steps = int(request.form.get('steps', 24))
        confidence_level = int(request.form.get('confidenceLevel', 95))
        print(f"接收到预测请求，预测类型: {predict_type}, 步数: {steps}, 置信区间: {confidence_level}")
        # 验证参数
        if predict_type not in ('single', 'multi'):
            raise ValueError("无效预测类型")
        if not (1 <= steps <= 72):
            raise ValueError("预测步数应在1-72范围内")
        if confidence_level not in (90, 95, 99):
            raise ValueError("不支持的置信区间")
        print(f"参数验证通过：")
        # 更新运行时配置
        with open(MODEL_CONFIG_PATH, 'r+') as f:
            config = json.load(f)
            print(f"过去模型配置：{config}")
            config['predict_type'].update({
                "type": predict_type,
                "steps": steps,
                "confidence": True,
                "confidence_level": confidence_level
            })
            print(f"更新模型配置：{config}")
            f.seek(0)
            json.dump(config, f, indent=4)
            f.truncate()
        print(f"模型配置更新成功")
        # 根据配置加载模型
        model = load_model_by_config(config)
        
        # 执行预测
        prediction_result = model.predict(
            steps=steps,
            confidence_level=confidence_level/100
        )

        return jsonify({
            'status': 'success',
            'prediction': prediction_result,
            'config': config
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

def load_model_by_config(config):
    model_name = config['model_name']
    if model_name == "Bayesian_LSTM":
        from models.bayesian_lstm import BayesianLSTMPredictor
        return BayesianLSTMPredictor(config)
    elif model_name == "CEEMDAN-LSTM":
        from models.ceemdan_lstm import CEEMDANLSTMPredictor
        return CEEMDANLSTMPredictor(config)
    else:
        raise ValueError("未知模型类型")
    

def get_latest_processed_data():
    """获取最新处理后的数据文件路径"""
    # 实现逻辑示例：
    # processed_dir = Path('data/processed')
    # files = sorted(processed_dir.glob('*.csv'), key=os.path.getmtime)
    # return str(files[-1]) if files else None
    return 'data/processed/latest.csv'  # 需要根据实际项目实现


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
