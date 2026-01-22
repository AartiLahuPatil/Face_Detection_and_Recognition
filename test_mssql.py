import pyodbc

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=face_db;"
    "UID=face_admin;"
    "PWD=Face@456;"
)


print("Connected Successfully!")
conn.close()
