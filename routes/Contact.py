
from flask import Blueprint, render_template, request, redirect, url_for, flash

from models.Contact import insert_contact, get_all_contacts, delete_contact, update_contact
from Utils.JWTtoken import token_required


contact_bp = Blueprint('contact', __name__)

@contact_bp.route('/contact', methods=['GET', 'POST'])
@token_required
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        if not name or not email or not message:
            flash('All fields are required!', 'error')
            return redirect(url_for('contact.contact'))

        insert_contact(name, email, message)
        flash('Contact submitted successfully!', 'success')
        return redirect(url_for('contact.contact'))

    contacts = get_all_contacts()
    return render_template('contact.html', contacts=contacts)



