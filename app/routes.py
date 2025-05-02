from flask import Blueprint, render_template, redirect, url_for,Flask,request,jsonify
from flask import send_from_directory
from werkzeug.utils import secure_filename
import os
from datetime import datetime 
# from app.forms import UploadForm, ModelConfigForm

app=Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录（app/）
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # 项目根目录（project/）

app.config.update({
    'UPLOAD_FOLDER': os.path.join(os.path.dirname(__file__), 'data/raw'),
    'ALLOWED_EXTENSIONS': {'csv'},  # 必须使用 app.config 存储
    'SECRET_KEY': 'your_secure_key_here',
    'MAX_CONTENT_LENGTH': 100 * 1024 * 1024
})


def allowed_file(filename):
    """ 正确的配置引用方式 """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/home',methods=['GET', 'POST'])
def home():
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': '未选择文件'}), 400
            
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'error': '文件类型不支持'}), 400
        print(f"当前文件路径: {os.path.abspath(__file__)}")
        print(f"项目根路径: {PROJECT_ROOT}")
        print(f"上传目录: {app.config['UPLOAD_FOLDER']}")
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        
        return jsonify({
            'status': 'success',
            'filename': filename,
            'preview_url': url_for('data_preprocessing')
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        },ascii=False), 500



@app.route('/dashboard')
def dashboard():
    if request=="POST":
        data=request.form
        print(data)
    # 模拟实时数据（后续可替换为真实数据源）
    wind_data = {
        'current_power': 15.8,
        'avg_wind_speed': 9.2,
        'capacity_factor': 0.68
    }
    return render_template('base.html',
                         title='Wind Power Forecast Dashboard',
                         current_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
                         wind_data=wind_data) 


@app.route('/upload', methods=['GET'])
def show_upload_page():
    return render_template('upload.html')  # 您的上传页面模板


# # 修改上传路由
@app.route('/upload', methods=['POST'])
def handle_file_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '未选择文件'}), 400
            
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'error': '文件类型不支持'}), 400
        print(f"当前文件路径: {os.path.abspath(__file__)}")
        print(f"项目根路径: {PROJECT_ROOT}")
        print(f"上传目录: {app.config['UPLOAD_FOLDER']}")
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        
        return jsonify({
            'status': 'success',
            'filename': filename,
            'preview_url': url_for('data_preprocessing')
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        },ascii=False), 500


# 在Flask路由中添加以下辅助函数
@app.route('/get-files')
def get_files():
    return jsonify({
        'raw': get_files_in_category('raw'),
        'processed': get_files_in_category('processed'),
        'predictions': get_files_in_category('predictions')
    })

def get_files_in_category(category):
    dir_path = os.path.join(BASE_DIR, 'data', category)
    if not os.path.exists(dir_path):
        return []
    return [{
        'name': f,
        'size': sizeof_fmt(os.path.getsize(os.path.join(dir_path, f))),
        'modified': datetime.fromtimestamp(os.path.getmtime(os.path.join(dir_path, f))).strftime('%Y-%m-%d %H:%M')
    } for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]





@app.route('/delete-files', methods=['POST'])
def delete_files():
    try:
        data = request.get_json()
        for file_info in data['files']:
            category_dir = os.path.join(BASE_DIR, 'data', file_info['category'])
            file_path = os.path.join(category_dir, file_info['name'])
            if os.path.exists(file_path):
                os.remove(file_path)
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/file-stats')
def get_file_stats():
    # 实现文件统计功能
    return jsonify({
        'raw': get_dir_stats('raw'),
        'processed': get_dir_stats('processed'),
        'predictions': get_dir_stats('predictions')
    })

def get_dir_stats(category):
    dir_path = os.path.join(BASE_DIR, 'data', category)
    files = []
    total_size = 0
    
    if os.path.exists(dir_path):
        for f in os.listdir(dir_path):
            fp = os.path.join(dir_path, f)
            if os.path.isfile(fp):
                size = os.path.getsize(fp)
                files.append({'name': f, 'size': sizeof_fmt(size)})
                total_size += size
                
    return {
        'count': len(files),
        'total_size': sizeof_fmt(total_size),
        'files': files
    }

def sizeof_fmt(num, suffix='B'):
    for unit in ['','K','M','G']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Ti{suffix}"    


@app.route('/preprocess')
def data_preprocessing():
    # 模拟预处理数据展示
    sample_data = {
        'before': [10, 12, 15, 11, 9],
        'after': [10, 12, 15, 11, 9]
    }
    return render_template('preprocessing.html', data=sample_data)

# @app.route('/train', methods=['GET', 'POST'])
# def model_training():
#     form = ModelConfigForm()
#     if form.validate_on_submit():
#         return redirect(url_for('training_progress'))
#     return render_template('train.html', form=form)

@app.route('/training-progress')
def training_progress():
    return render_template('training_progress.html')

@app.route('/predict')
def prediction():
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True) 