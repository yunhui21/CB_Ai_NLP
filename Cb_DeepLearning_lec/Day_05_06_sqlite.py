# Day_05_06_sqlite.py
import sqlite3
import Day_04_04_csv


# CRUD
# SQL : Structured Query Language
def create_db():
    conn = sqlite3.connect('data/weather.sqlite3')
    cur = conn.cursor()

    # CREATE TABLE db_list (id INTEGER, name VARCHAR(16))
    query = 'CREATE TABLE kma (prov TEXT, city TEXT, mode TEXT, tmEf TEXT, wf TEXT, tmn TEXT, tmx TEXT, rnSt TEXT)'
    cur.execute(query)

    conn.commit()
    conn.close()


def insert_row(row):
    conn = sqlite3.connect('data/weather.sqlite3')
    cur = conn.cursor()

    # INSERT INTO db_list (id, name) VALUES(1, "PC")
    base = 'INSERT INTO kma VALUES ("{}", "{}", "{}", "{}", "{}", "{}", "{}", "{}")'
    query = base.format(row[0], row[1], row[2], row[3],
                        row[4], row[5], row[6], row[7])
    cur.execute(query)

    conn.commit()
    conn.close()


def insert_all(rows):
    conn = sqlite3.connect('data/weather.sqlite3')
    cur = conn.cursor()

    # INSERT INTO db_list (id, name) VALUES(1, "PC")
    base = 'INSERT INTO kma VALUES ("{}", "{}", "{}", "{}", "{}", "{}", "{}", "{}")'
    for row in rows:
        # query = base.format(row[0], row[1], row[2], row[3],
        #                     row[4], row[5], row[6], row[7])
        query = base.format(*row)
        cur.execute(query)

    conn.commit()
    conn.close()


def fetch_all():
    conn = sqlite3.connect('data/weather.sqlite3')
    cur = conn.cursor()

    # SELECT * FROM db_list
    query = 'SELECT * FROM kma'
    for row in cur.execute(query):
        print(row)

    # conn.commit()
    conn.close()


def search_city(city):
    conn = sqlite3.connect('data/weather.sqlite3')
    cur = conn.cursor()

    # SELECT * FROM db_list WHERE id = 3
    # query = 'SELECT * FROM kma WHERE city = "' + city + '"'
    query = 'SELECT * FROM kma WHERE city = "{}"'.format(city)
    for row in cur.execute(query):
        print(row)

    # conn.commit()
    conn.close()


# create_db()

# rows = Day_04_04_csv.read_csv_3('data/weather.csv')
# print(*rows, sep='\n')

# for row in rows:
#     insert_row(row)

# insert_all(rows)

# fetch_all()

# 문제
# 전달한 city와 같은 도시의 날씨만 보여주세요
search_city('청주')
