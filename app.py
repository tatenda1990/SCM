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
from scm_calc import plot_recoveries, return_df,best_model,plot_linear_kinetic_selected_models
from scm_calc import surface_chem,diffusion,mixed,film

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
        recoveries_url = plot_recoveries(destination)
        df = return_df(destination)
        selected_models = [surface_chem,diffusion,mixed,film]
        model = best_model(df,selected_models)
        _url = plot_linear_kinetic_selected_models ([model[6]])
        model_matrix = model[0]
        best_model_params = {
            'name' : model[1],
            'r2' : model[2],
            'best_model_slope': model[3],
            'best_model_A': model[4],
            'best_model_Ea': model[5],
            'arrhenius_url': model[7]}
       
        return render_template('results.html', 
                               posts = posts, 
                               form = form, 
                               recoveries_url = recoveries_url,
                               df = [df.to_html(classes = 'data')],
                               model_matrix = [model_matrix.to_html(classes = 'data')],
                               best_model_params = best_model_params,
                               _url = _url,
                               titles = df.columns.values)
    return render_template('home.html', posts = posts, form = form)

@app.route("/about")
def about ():
    return (render_template('about.html'))

if __name__ == '__main__':
    app.run(debug = True)
