from flask import Flask,request, render_template

from tools.data_manager import *


app = Flask(__name__)
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_files():
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