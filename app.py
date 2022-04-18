from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Declare a Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        clf = joblib.load("clf.pkl")
        
        # Get values through input bars
        height = request.form.get("height")
        gender = request.form.get("gender")
        
        # Put inputs to dataframe
        
        X = pd.DataFrame([[height, gender]], columns = ["Height", "Gender"])
        le = LabelEncoder()
        X["Gender"] = le.fit_transform(X["Gender"])
        # Get prediction
        prediction = clf.predict(X)[0]
        
    else:
        prediction = ""
        
    return render_template("website.html", output = prediction)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)