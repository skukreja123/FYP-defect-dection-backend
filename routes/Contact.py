
from flask import Blueprint, render_template, request, redirect, url_for, flash

from models.Contact import insert_contact, get_all_contacts
from Utils.JWTtoken import token_required


contact_bp = Blueprint('contact', __name__)

@contact_bp.route('/contact', methods=['GET', 'POST'])
@token_required
def contact():
    if request.method == 'POST':
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        message = data.get('message')

        if not name or not email or not message:
            return {'error': 'All fields are required!'}, 400

        insert_contact(name, email, message)
        return {'message': 'Contact submitted successfully!'}, 200

    contacts = get_all_contacts()
    return render_template('contact.html', contacts=contacts)


