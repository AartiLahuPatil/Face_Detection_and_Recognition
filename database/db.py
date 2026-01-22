import pyodbc
import pickle
import numpy as np

CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=face_db;"
    "UID=face_admin;"
    "PWD=Face@456;"
)

def save_embedding(name, embedding):
    conn = pyodbc.connect(CONN_STR)
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO users (name, embedding) VALUES (?, ?)",
        (name, pyodbc.Binary(pickle.dumps(embedding)))
    )

    conn.commit()
    conn.close()

def load_embeddings():
    conn = pyodbc.connect(CONN_STR)
    cur = conn.cursor()

    cur.execute("SELECT name, embedding FROM users")
    rows = cur.fetchall()
    conn.close()

    names, embeddings = [], []

    for row in rows:
        names.append(row[0])
        embeddings.append(pickle.loads(bytes(row[1])))

    return names, np.array(embeddings)
