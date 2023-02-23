from flask import Flask,render_template,request

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def home():
    data = request.form.get('input','')
    return render_template('app.html',output=data)

# @app.route('/api',methods=['GET','POST'])
# def api():
#     return request.form['input']