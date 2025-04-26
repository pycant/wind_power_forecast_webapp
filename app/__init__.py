from .routes import bp
from config import settings

def create_app():
    from flask import Flask
    app = Flask(__name__)
    app.config.from_object(settings)
    app.config["SECRET_KEY"] = "secretkey"
    app.register_blueprint(bp)
    return app

app = create_app()