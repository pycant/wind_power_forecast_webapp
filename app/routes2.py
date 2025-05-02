from flask import Flask, render_template, request, jsonify, redirect, url_for,render_template
import os
import uuid
from werkzeug.utils import secure_filename
import data_manager as dm
from data_manager import *
from utils.data_cleaner import *
import unicodedata
import re
import json

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
    return render_template('base.html')



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

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
