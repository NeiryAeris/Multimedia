import sqlite3
import os

# DB_PATH = "../Database/structured_features_ver2.db"
DB_PATH = './Database/features.db'

# Check if file exists first
if os.path.exists(DB_PATH):
    try:
        conn = sqlite3.connect(DB_PATH)
        print("Connected successfully!")
        conn.close()
    except sqlite3.Error as e:
        print("Failed to connect:", e)
else:
    print(f"Database file '{DB_PATH}' does not exist.")

# import os
# print("Working directory:", os.getcwd())

