from flask import Flask, render_template, request
from modules import generate_text, preprocess, getPositionEncoding

app = Flask(__name__)
data = ''
output = ''

@app.route('/', methods=['GET', 'POST'])
def home():
  global data
  global output
  color=''
  output=''
  data = request.form.get('input', '')
  if data != '':
    output,color = generate_text(data)
  return render_template('app.html', input=data, output=output,color=color)


@app.route('/api', methods=['GET', 'POST'])
def api():
  if output != "INTERNAL ERROR!":
    o1, o2, o3, o4, o5 = preprocess(data)
    o6 = getPositionEncoding(len(list(data.split(' '))))
  else:
    o1, o2, o3, o4, o5, o6 = 'ERR','ERR','ERR','ERR','ERR','ERR'
  return render_template('api.html', o1=o1, o2=o2, o3=o3, o4=o4, o5=o5, o6=o6)


if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True)
