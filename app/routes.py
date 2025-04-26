from flask import Blueprint, render_template, redirect, url_for,Flask
from app.forms import UploadForm, ModelConfigForm

bp = Blueprint('main', __name__)

# app=Flask(__name__)
# app.config["SECRET_KEY"]="secretkey"

@bp.route('/')
def dashboard():
    return render_template('index.html', 
                         title='Wind Power Forecast Dashboard')

@bp.route('/upload', methods=['GET', 'POST'])
def data_upload():
    form = UploadForm()
    if form.validate_on_submit():
        # 模拟文件保存逻辑
        return redirect(url_for('data_preprocessing'))
    return render_template('upload.html', form=form)

@bp.route('/preprocess')
def data_preprocessing():
    # 模拟预处理数据展示
    sample_data = {
        'before': [10, 12, 15, 11, 9],
        'after': [10, 12, 15, 11, 9]
    }
    return render_template('preprocess.html', data=sample_data)

@bp.route('/train', methods=['GET', 'POST'])
def model_training():
    form = ModelConfigForm()
    if form.validate_on_submit():
        return redirect(url_for('training_progress'))
    return render_template('train.html', form=form)

@bp.route('/training-progress')
def training_progress():
    return render_template('training_progress.html')

@bp.route('/predict')
def prediction():
    return render_template('predict.html')

# if __name__ == '__main__':
#     app.run(debug=True) 