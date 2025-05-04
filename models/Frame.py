from db import get_db_connection

def create_frame_table():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS frames (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            frame_data BYTEA NOT NULL,
            keras_label TEXT,
            keras_confidence FLOAT,
            pytorch_label TEXT,
            pytorch_confidence FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)
    conn.commit()
    cur.close()
    conn.close()


def insert_frame_with_predictions(user_id, frame_data, keras_label, keras_confidence, pytorch_label, pytorch_confidence):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO frames (
            user_id, frame_data, keras_label, keras_confidence, pytorch_label, pytorch_confidence
        ) VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (user_id, frame_data, keras_label, keras_confidence, pytorch_label, pytorch_confidence))
    frame_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return frame_id


def get_frame_by_id(frame_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM frames WHERE id=%s", (frame_id,))
    frame = cur.fetchone()
    cur.close()
    conn.close()
    return frame

def get_all_frames_by_user_id(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM frames WHERE user_id=%s", (user_id,))
    frames = cur.fetchall()
    cur.close()
    conn.close()
    return frames

def delete_frame_by_id(frame_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM frames WHERE id=%s", (frame_id,))
    conn.commit()
    cur.close()
    conn.close()
    





    