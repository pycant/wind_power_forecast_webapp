from flask import Flask, render_template, request, jsonify
import os
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'csv'}

# 初始化上传目录
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # 获取前端交互元素信息
    button_clicked = request.form.get('button_id')
    user_agent = request.headers.get('User-Agent')
    
    # 实时输出用户操作日志
    print(f"User operation detected - Button: {button_clicked}, Client: {user_agent}")

    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': '未选择文件'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': '未选择文件'}), 400

    try:
        # 前端验证反馈
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'validation_error',
                'errors': ['仅支持CSV格式文件']
            }), 400

        # 文件重名检查
        filename = secure_filename(file.filename)
        if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
            return jsonify({
                'status': 'validation_error',
                'errors': ['文件名已存在，请修改后重新上传']
            }), 400

        # 保存文件
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # 记录成功日志
        print(f"文件上传成功：{filename}，大小：{os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], filename))}字节")
        
        return jsonify({
            'status': 'success',
            'filename': filename,
            'preview_url': '/preview?file=' + filename
        }), 200

    except Exception as e:
        # 错误处理及实时反馈
        error_message = f"文件上传失败：{str(e)}"
        print(error_message)
        return jsonify({
            'status': 'error',
            'message': error_message
        }), 500

@app.route('/preview')
def data_preview():
    # 获取文件预览参数
    filename = request.args.get('file')
    page = request.args.get('page', 1, type=int)
    per_page = 20  # 每页显示20条
    
    # 实现分页逻辑
    # 这里添加具体的数据读取和分页处理代码
    
    return jsonify({
        'status': 'preview_ready',
        'filename': filename,
        'page': page,
        'per_page': per_page
    })

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
