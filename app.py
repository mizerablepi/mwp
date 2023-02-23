import os
from flask import Flask,render_template,request
import openai

my_secret = os.environ['KEY']
openai.api_key = my_secret

def generate_text(prompt):
    completions = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt+"show with an equation",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.0,
        top_p=1.0,
        presence_penalty=-0.6,
    )

    message = completions.choices[0].text
    return message.strip()
  
app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def home():
    output = ''
    data = request.form.get('input',None)
    if data != None:
      output = generate_text(data)
    return render_template('app.html',output=output)

# @app.route('/api',methods=['GET','POST'])
# def api():
#     return request.form['input']
if __name__ == '__main__':
  app.run(host='0.0.0.0',debug=True)