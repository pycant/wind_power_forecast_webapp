from flask import Flask, request, render_template, jsonify, redirect, url_for
import os
from werkzeug.utils import secure_filename
from data_manager import * 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/file-manager', methods=['GET','POST'])
def file_manager():
    file_stats=get_file_stats()
    if request.method == 'POST':
        print("接收到文件上传请求", request.files, request.form,sep='\n')
    
    if request.method == 'GET':
        print("渲染文件管理页面", file_stats,file_stats['raw']['count'],sep='\n')
        print()
        return render_template('data_file_manger.html',file_stats=file_stats)
if __name__ == '__main__':
    app.run(debug=True)

