# Day_10_05_flask.py
from flask import Flask, render_template
import random

# static, templates


app = Flask(__name__)


@app.route('/')
def index():
    return '쉬는 시간입니다'


@app.route('/lotto')
def lotto():
    numbers = [random.randrange(45) + 1 for _ in range(6)]
    return str(numbers)


@app.route('/html')
def html():
    numbers = [random.randrange(45) + 1 for _ in range(6)]
    return render_template('randoms.html',
                           numbers=numbers)


if __name__ == '__main__':
    app.run(debug=True)

