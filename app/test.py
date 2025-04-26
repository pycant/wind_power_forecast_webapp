from flask import Flask, request, render_template, jsonify, redirect, url_for
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('base.html')

# 处理文件上传
@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({
            'status': 'success',
            'filename': filename,
            'message': 'File uploaded successfully'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/upload', methods=['GET'])
def load_upload_page():
    return render_template('upload.html')

# 处理预处理表单
@app.route('/data_preprocessing_submit', methods=['POST'])
def handle_preprocessing():
    features = request.form.getlist('features')
    print("选择的预处理特征：", features)
    return redirect(url_for('index') + '#preprocess')

# 处理模型选择
@app.route('/model_selection_submit', methods=['POST'])
def handle_model_selection():
    print("选择的模型：CEEMDAN-LSTM")
    return redirect(url_for('index') + '#model')

# 处理预测提交
@app.route('/predict_submit', methods=['POST'])
def handle_prediction():
    predict_type = request.form.get('predict-type')
    print("预测类型：", predict_type)
    return redirect(url_for('index') + '#predict')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'csv'}

if __name__ == '__main__':
    app.run(debug=True)