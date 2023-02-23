from flask import Flask,render_template

app = Flask(__name__)

render_template('app.html')
