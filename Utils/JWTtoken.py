import jwt
from datetime import datetime, timedelta
from config import Config

def generate_token(email):
    payload = {
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
    def decorator(*args, **kwargs):
        token = kwargs.get('token')
        if not token:
            return {'message': 'Token is missing!'}, 401

        try:
            data = decode_token(token)
            if not data:
                return {'message': 'Token is invalid or expired!'}, 401
        except Exception as e:
            return {'message': str(e)}, 401

        return f(*args, **kwargs)

    return decorator