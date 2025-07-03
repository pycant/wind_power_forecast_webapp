from tools.uploads import *
from flask import Flask,request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    return upload_files()

if __name__ == '__main__':
    app.run(debug=True)