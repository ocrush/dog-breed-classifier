from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from dog_classifier import DogPredictor
import os

UPLOAD_FOLDER = 'static/uploads/'

dog_classifier = DogPredictor()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def upload_img():
    return render_template('master.html')


# index webpage displays cool visuals and receives user input text for model
@app.route('/', methods=['POST'])
@app.route('/index', methods=['POST'])
def index():
    if request.method == 'POST':
        if 'imgFile' not in request.files:
            flash('No file part')
            return redirect(request.url)
        img_file = request.files['imgFile']
        if img_file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        # only image files are accepted so can proceed here
        filename = secure_filename(img_file.filename)
        img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        matching_breed, error = dog_classifier.dog_breed_matching(
            os.path.join(app.config['UPLOAD_FOLDER'], filename))
        if len(error) > 0:
            return render_template("master.html", orig_image=os.path.join("uploads", filename), error=error)

        return render_template("master.html",
                               orig_image=os.path.join("uploads", filename),
                               matching_breed=os.path.join("predicted", matching_breed + ".jpg"),
                               breed_name=matching_breed)


# web page that handles user query and displays model results
@app.route('/results/<filename>')
def results(filename):
    # This will render the go.html Please see that file.
    image_file = "uploads/" + filename
    return render_template(
        'results.html',
        image_file=image_file
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
