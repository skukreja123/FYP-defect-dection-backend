import jwt
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify
from config import Config

def generate_token(email, user_id):
    payload = {
        "user_id": user_id,
        'email': email,
        'exp': datetime.utcnow() + timedelta(hours=2)
    }
    return jwt.encode(payload, Config.SECRET_KEY, algorithm='HS256')

def decode_token(token):
    try:
        return jwt.decode(token, Config.SECRET_KEY, algorithms=['HS256'])
    except jwt.ExpiredSignatureError:
        return None

def token_required(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]

        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        try:
            data = decode_token(token)
            print(f"Decoded token data: {data}")  # Log decoded data for debugging
            current_user = data['user_id'] if data else None
            if not data:
                return jsonify({'message': 'Token is invalid or expired!'}), 401
        except Exception as e:
            return jsonify({'message': str(e)}), 401

        # Attach decoded data (like email) to request context if needed
        print(f"Current user ID: {current_user}")
        return f(current_user, *args, **kwargs)

    return decorator