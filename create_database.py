import psycopg2

conn = psycopg2.connect(
    host='100.100.224.121',
    port=5433,
    user='postgres',
    password='admin',
    dbname='postgres'
)
conn.autocommit = True

cur = conn.cursor()
cur.execute('CREATE DATABASE "freedom-model"')

print('Database "freedom-model" created successfully.')
conn.close()
