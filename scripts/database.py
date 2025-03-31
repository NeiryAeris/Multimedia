import sqlite3
import json
from config import DB_PATH, JSON_OUTPUT
import numpy as np

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS image_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            label TEXT,
            color_histogram TEXT,
            shape_descriptor TEXT,
            texture_descriptor TEXT,
            deep_embedding TEXT
        )
    """)
    return conn, c

def insert_features(cursor, data):
    cursor.executemany("""
        INSERT INTO image_features (
            image_path, label, color_histogram, shape_descriptor, texture_descriptor, deep_embedding
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, data)

def save_sample(cursor):
    cursor.execute("SELECT * FROM image_features LIMIT 10;")
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    with open(JSON_OUTPUT, "w") as f:
        json.dump([dict(zip(columns, row)) for row in rows], f, indent=4)

def load_database_features(DB_PATH = "./Database/structured_features_ver2.db"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT image_path, color_histogram, shape_descriptor, texture_descriptor, deep_embedding FROM image_features")
    rows = cursor.fetchall()
    conn.close()

    image_paths = []
    color_features = []
    shape_features = []
    texture_features = []
    deep_features = []

    for row in rows:
        image_paths.append(row[0])
        color_features.append(json.loads(row[1]))
        shape_features.append(json.loads(row[2]))
        texture_features.append(json.loads(row[3]))
        deep_features.append(json.loads(row[4]))

    return image_paths, np.array(color_features), np.array(shape_features), np.array(texture_features), np.array(deep_features)
