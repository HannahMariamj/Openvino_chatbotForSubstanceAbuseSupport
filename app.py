from flask import Flask, render_template, request, jsonify
from load_model_for_inference import ov_model, tokenizer
from rag_module import final_result

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    if not msg:
        return jsonify({"error": "No message provided"}), 400
    

    print("User input:", msg)
    result = final_result(msg, ov_model, tokenizer)
    print("Response:", result)

    return jsonify({"response": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
