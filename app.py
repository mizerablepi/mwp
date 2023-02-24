from flask import Flask, render_template, request
from modules import generate_text

app = Flask(__name__)
data = ''


@app.route('/', methods=['GET', 'POST'])
def home():
  output = ''
  global data
  data = request.form.get('input', '')
  if data != '':
    output = generate_text(data)
  return render_template('app.html', input=data, output=output)


@app.route('/api', methods=['GET', 'POST'])
def api():
  #TODO: preprocessing
  return render_template('api.html')


if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True)
