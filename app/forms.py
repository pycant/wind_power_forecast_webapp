from flask_wtf import FlaskForm

from wtforms import (FileField, SelectField, IntegerField, 
                    SubmitField, BooleanField)
from wtforms.validators import DataRequired

class UploadForm(FlaskForm):
    data_file = FileField('Data File', validators=[
        DataRequired(message="请选择要上传的文件"),
        FileAllowed(['csv'], message="仅支持CSV格式文件")
    ])
    submit = SubmitField('上传')

class ModelConfigForm(FlaskForm):
    model_type = SelectField('Model Type', choices=[
        ('lstm', 'Bayesian LSTM'),
        ('rf', 'Random Forest'),
        ('hybrid', 'Hybrid Model')
    ])
    time_steps = IntegerField('Time Steps', default=24)
    enable_gpu = BooleanField('Enable GPU Acceleration')
    submit = SubmitField('Start Training')