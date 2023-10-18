from fastai import *
from fastai.vision import *
from PIL import Image
import pickle
from fastai.vision.all import *
from PIL import Image 
from flask import Flask, request, jsonify
from flask_cors import CORS
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    try:

 
        image_file = request.files["image"]
        unique_filename = "IMG" + ".jpg"
        image_file.save(unique_filename)
        
        learn = pickle.load(open("exportFinal.pkl", "rb"))
        image_path = "IMG.jpg"
                
        img = PILImage.create(image_path)
        prediction = learn.predict(img)
        # Get the predicted class label
        predicted_class = prediction[0]
        probabilities = prediction[2]

        return jsonify({"predicted_quality": predicted_class})
    except Exception as e:
            print(e)
            return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
