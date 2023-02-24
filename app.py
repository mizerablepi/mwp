import os
from flask import Flask,render_template,request
from modules import test
#import openai

# def generate_text(prompt):
    # my_secret = os.environ['KEY']
    # openai.api_key = my_secret
    # completions = openai.Completion.create(
        # engine="text-davinci-003",
        # prompt=prompt+" show with an equation",
        # max_tokens=1024,
        # n=1,
        # stop=None,
        # temperature=0.0,
        # top_p=1.0,
        # presence_penalty=-0.6,
    # )
# 
    # message = completions.choices[0].text
    # return message.strip()
  
app = Flask(__name__)
data = ''
@app.route('/', methods=['GET','POST'])
def home():
    output = ''
    global data 
    data = request.form.get('input','')
    #if data != None:
      #output = generate_text(data)
    return render_template('app.html',output=data)

@app.route('/api',methods=['GET','POST'])
def api():
    new = test(data)
    return render_template('api.html',out1=new,out2=data+'second',out3=data+'third') 

if __name__ == '__main__':
  app.run(host='0.0.0.0',debug=True)