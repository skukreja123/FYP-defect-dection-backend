from flask import Blueprint, request, jsonify
import bcrypt
from models.User import get_user_by_email, insert_user
from Utils.JWTtoken import generate_token, decode_token

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if get_user_by_email(email):
        return jsonify({'error': 'User already exists'}), 409

    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    user_id = insert_user(email, hashed_pw)
    return jsonify({'message': 'User created', 'user_id': user_id}), 201

@auth_bp.route('/signin', methods=['POST'])
def signin():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = get_user_by_email(email)
    if user and bcrypt.checkpw(password.encode('utf-8'), user[2].encode('utf-8')):
        token = generate_token(email,user[0])
        return jsonify({'token': token}), 200
    return jsonify({'error': 'Invalid credentials'}), 401

@auth_bp.route('/signout', methods=['POST'])
def signout():
    # For stateless JWT, sign-out can be done client-side by deleting token
    return jsonify({'message': 'Signed out'}), 200
