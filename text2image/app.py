from flask import Flask, request, render_template
import tensorflow as tf
import random
from utils import contains_only_non_alphanumeric_chars
import matplotlib.pyplot as plt
from PIL import Image
from image_generator import run

# from models.embedding_generator import Embedder
# from models.generator import build_generator


app = Flask(__name__)



# home page
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/display_image', methods = ['GET','POST'])
def image():
    random_number = []

    if request.method == 'POST':
        # Get text
        text = request.form.get("text")
        samples = int(request.form.get("samples"))

        # # Validate input text
        # if (contains_only_non_alphanumeric_chars(text)) == False:
        #     error_msg = "Please enter Nepalese characters"
        #     return render_template("error.html", error_msg = error_msg)
        # elif (text == ""):
        #     error_msg = "Cannot take empty text input"
        #     return render_template("error.html", error_msg = error_msg)
        
        for i in range(samples):
            random_number.append(random.randint(0,100))


        # passing noise and embeddings to trained generator
        generated_image = run(text, random_number=random_number,samples = samples)

        return render_template('display_image.html', number = random_number, caption=text)
    else:
        return render_template('display_image.html')
    

if __name__ == '__main__':
    app.run()

