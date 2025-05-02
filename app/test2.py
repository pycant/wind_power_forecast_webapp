import os
from flask import render_template, request, jsonify, send_from_directory, Flask
from werkzeug.utils import secure_filename
from datetime import datetime
app=Flask(__name__)
# 文件管理配置
DATA_BASE = os.path.join(os.path.dirname(__file__), 'data')
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_info(filepath):
    stat = os.stat(filepath)
    return {
        'name': os.path.basename(filepath),
        'size': f"{stat.st_size / 1024 / 1024:.1f}MB",
        'date': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
        'path': filepath
    }
    
    



# 文件管理路由
@app.route('/file-manager')
def file_manager():
    return render_template('data_file_manager.html')

@app.route('/api/files', methods=['GET'])
def get_files():
    category = request.args.get('category', 'raw')
    valid_categories = ['raw', 'processed', 'predictions']
    
    if category not in valid_categories:
        return jsonify({'error': 'Invalid category'}), 400

    target_dir = os.path.join(DATA_BASE, category)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    files = []
    for filename in os.listdir(target_dir):
        filepath = os.path.join(target_dir, filename)
        if os.path.isfile(filepath):
            files.append(get_file_info(filepath))

    return jsonify({'files': files})


@app.route('/api/upload', methods=['POST'])
def upload_file():
    category = request.form.get('category', 'raw')
    file = request.files.get('file')
    
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file.content_length > MAX_FILE_SIZE:
        return jsonify({'error': 'File size exceeds 100MB limit'}), 413
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only CSV files allowed'}), 400

    filename = secure_filename(file.filename)
    target_dir = os.path.join(DATA_BASE, category)
    
    try:
        os.makedirs(target_dir, exist_ok=True)
        file.save(os.path.join(target_dir, filename))
        return jsonify({'message': 'File uploaded successfully', 'filename': filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/files', methods=['DELETE'])
def delete_file():
    data = request.get_json()
    category = data.get('category')
    filename = data.get('filename')
    
    if not all([category, filename]):
        return jsonify({'error': 'Missing parameters'}), 400
    
    target_path = os.path.join(DATA_BASE, category, filename)
    
    if not os.path.exists(target_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        os.remove(target_path)
        return jsonify({'message': 'File deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/data/<category>/<filename>')
def download_file(category, filename):
    valid_categories = ['raw', 'processed', 'predictions']
    if category not in valid_categories:
        return jsonify({'error': 'Invalid category'}), 400
    
    directory = os.path.join(DATA_BASE, category)
    return send_from_directory(directory, filename, as_attachment=True)


from flask import request, render_template
from data_manager import get_file_stats, delete_file, save_uploaded_file

def configure_routes(app):
    @app.route('/file-manager')
    def file_manager():
        stats = get_file_stats()
        return render_template('data_file_manger.html', file_stats=stats)

    @app.route('/api/files', methods=['DELETE'])
    def handle_file_operation():
        data = request.get_json()
        success, message = delete_file(data['path'])
        return jsonify({'success': success, 'message': message})

    @app.route('/api/upload', methods=['POST'])
    def handle_file_upload():
        category = request.form.get('category', 'raw')
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '未选择文件'})
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': '无效文件名'})
            
        success, result = save_uploaded_file(file, category)
        if success:
            return jsonify({'success': True, 'filename': result})
        else:
            return jsonify({'success': False, 'message': result})


if __name__ == '__main__':
    app.run(debug=True, port=5000)