from flask import Flask, render_template, request

from object import detectObjects

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    results = detectObjects(image_path)

    return render_template('index.html', prediction=results)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
