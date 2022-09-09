from flask import Flask,render_template
from wtforms import FileField, SubmitField
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scm_calculator import plot_recoveries, return_df

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'
#app.config['UPLOAD-FOLDER'] = 'static/files'

posts = [{'title': "Post 1",
          'author': "Collins",
          'date': "22 August 2022",
          'postcontent': "This is post 1"}]

class UnploadFileForm(FlaskForm):
    file = FileField('file', validators=[InputRequired()])
    submit = SubmitField('Upload File')

def func1(f):
    print(f.r)
    #df = pd.read_csv(f)
    #print (df)
    print("It's finally working")
    
@app.route("/", methods = ['GET', 'POST'])
@app.route("/home", methods = ['GET', 'POST'])
def home ():
    form = UnploadFileForm()
    if form.validate_on_submit():
        f = form.file.data
        destination = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],secure_filename(f.filename))
        f.save(destination)
        dummy_ = plot_recoveries(destination)
        df = return_df(destination)
        return render_template('home.html', posts = posts, form = form, plot_url = dummy_,df = [df.to_html(classes = 'data')], titles = df.columns.values)
    return render_template('home.html', posts = posts, form = form)

@app.route("/about")
def about ():
    return (render_template('about.html'))

if __name__ == '__main__':
    app.run(debug = True)
