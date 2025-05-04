from flask import Flask, jsonify
from flask_cors import CORS
from routes.video import video_bp
from routes.image import image_bp
from routes.auth import auth_bp
from models.User import create_user_table
from models.Contact import create_contact_table
from routes.Contact import contact_bp
from config import Config

app = Flask(__name__)
app.config.from_object('config.Config')
CORS(app)

create_user_table()
create_contact_table()
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask server is running."})


# Register Blueprints
app.register_blueprint(video_bp, url_prefix='/video')
app.register_blueprint(image_bp, url_prefix='/image')
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(contact_bp, url_prefix='/contact')


if __name__ == '__main__':
    port = Config.PORT
    app.run(host="0.0.0.0", port=port,threaded=True)
