from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/process_json', methods=['POST'])
def process_json():
    # Check if the request contains JSON data
    if request.is_json:
        # Extract JSON data from the request
        data = request.get_json()

        # Process the JSON data (example: just echoing back the received data)
        print(type(data['user_message']))

        #print(processed_data)

        # Return the processed data as JSON
        return jsonify(data), 200
    else:
        # If the request does not contain JSON data, return an error
        error_response = {"error": "Request must contain JSON data"}
        return jsonify(error_response), 400

if __name__ == '__main__':
    app.run(debug=True)
