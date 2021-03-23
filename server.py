#get array of data from postman post ///// set headers :key,value , set url for POST // inset json body ["Kelvin","maria"]

# from flask import Flask, request, jsonify
# from gender_api import GenderAPI

# # Init Flask and GenderAPI
# app = Flask(__name__)
# api = GenderAPI()

# @app.route('/')
# def hello_world():
#     return "Welcome to GenderAPI. Please request on /predict using \
#     'Content-Type: application/json' header and a json array of names in the body."

# @app.route('/predict', methods=['POST'])
# def predict():
#     names = request.get_json()
#     print(names)
#     labels = api.predict(names)
#     print(labels)
#     return jsonify(labels)

# # Run Flask
# app.run(host='127.0.0.1', port=5000, debug=False)

#get array of data from HTML form /// example  "Kelvin","maria"



from flask import Flask, request,redirect, url_for,  jsonify,render_template
from gender_api import GenderAPI

# Init Flask and GenderAPI
app = Flask(__name__)
api = GenderAPI()


@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])

# def my_form_post():
#     text = request.form['text']
#     processed_text = text.upper()
#     return processed_text

def predict():
    names = request.form['text']
    print(str(names))
    names=str(names).split(",")
    print(names)
    labels = api.predict(names)
    return jsonify(labels)



app.run(host='0.0.0.0', port=4000, debug=False)


