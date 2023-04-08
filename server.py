from flask import Flask, jsonify, request
from flask_cors import CORS


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3002"}})

data = [6,7,8,9,10]

@app.route('/')
def base():
    return "Hello world, this is flask"
@app.route('/get', methods= ['GET'])
def get():
    output = {'arr': data, 'msg': "GET REQUEST FROM FLASK"}
    return jsonify(output)
@app.route('/post', methods=['POST'])
def post_data():
    data = request.get_json() # retrieve the JSON data from the request
    numbers = data['data'] # retrieve the 'data' key from the JSON data
    output = [num * 2 for num in numbers] # multiply each number by 2
    response_data = {'arr': ["P", "O", "S", "T"], 'message': "POST REQUEST FROM FLASK"}
    return jsonify(response_data)



if __name__ == '__main__':
    app.run(debug=True)