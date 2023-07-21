# https://towardsdatascience.com/building-a-machine-learning-web-application-using-flask-29fa9ea11dac
from flask import Flask, request, render_template
import pandas as pd
import pickle
import os

species_decoding = {
    0 : "Iris-setosa",
    2 : "Iris-virginica" ,
    1 : "Iris-versicolor"
}

# Declare a Flask app
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        filename = "./models/knn_model.sav"
        knn = pickle.load(open(filename, 'rb'))

        filename = "./models/svc_model.sav"
        svc = pickle.load(open(filename, 'rb'))
        
        # Get values through input bars
        sep_len = request.form.get("sepal_length")
        sep_wid = request.form.get("sepal_width")
        pet_len = request.form.get("petal_length")
        pet_wid = request.form.get("petal_width")

        inp_vals = [float(sep_len), float(sep_wid), float(pet_len), float(pet_wid)]

        data = pd.DataFrame([inp_vals], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

        # Calculate other features
        data["sepal_area"] = data["sepal_length"] * data["sepal_width"]
        data["petal_area"] = data["petal_length"] * data["petal_width"]
        
        # Get prediction
        prediction_knn = f"KNN prediction: {species_decoding[knn.predict(data)[0]]}"
        prediction_svc = f"SVC prediction: {species_decoding[svc.predict(data)[0]]}"
        
    else:
        prediction_knn = ""
        prediction_svc = ""
        
    return render_template("index.html", knn = prediction_knn, svc = prediction_svc)

if __name__ == '__main__':
    app.run(debug = True)