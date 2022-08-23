from flask import Flask, render_template
app = Flask(__name__)

posts = [                                           #assume this is a call to a database 
        {'author':'Collins Saguru',
          'title':'Blog Post 1',
          'content':'First Post Content',
          'date':'April 21,2018'
         },
         {'author':'Tatenda Saguru',
          'title':'Blog Post 2',
          'content':'Second Post Content',
          'date':'April 26,2018'
         }
         ]

@app.route("/")
@app.route("/home")

def home ():
    return render_template('home.html', posts = posts)

@app.route("/about")
def about ():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug = True)