from db import get_db_connection

def create_contact_table():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            message TEXT NOT NULL,
        )
    """)
    conn.commit()
    cur.close()
    conn.close()
    
def get_contact_by_email(email):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM contacts WHERE email=%s", (email,))
    contact = cur.fetchone()
    cur.close()
    conn.close()
    return contact

def insert_contact(name, email, message):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO contacts (name, email, message) VALUES (%s, %s, %s) RETURNING id", (name, email, message))
    contact_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return contact_id

def get_all_contacts():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM contacts")
    contacts = cur.fetchall()
    cur.close()
    conn.close()
    return contacts

