import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify

# --- FLASK APP INITIALIZATION ---
app = Flask(__name__)

# -----------------------------------------------------
# LOAD PREPROCESSOR CONFIG
# -----------------------------------------------------
try:
    with open("preprocessor_config.json", "r") as f:
        preproc = json.load(f)
except FileNotFoundError:
    print("Error: preprocessor_config.json not found.")
    exit()

IMAGE_SIZE = preproc["size"]["height"]
MEAN = np.array(preproc["image_mean"], dtype="float32")
STD = np.array(preproc["image_std"], dtype="float32")
RESCALE = preproc["rescale_factor"]

# -----------------------------------------------------
# LOAD MODEL CONFIG (id2label)
# -----------------------------------------------------
try:
    with open("config.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    print("Error: config.json not found.")
    exit()

id2label = {int(k): v for k, v in config["id2label"].items()}

# -----------------------------------------------------
# PREPROCESSING FUNCTION
# This function is modified to accept a PIL Image object instead of a path.
# -----------------------------------------------------
def preprocess_image(img_obj: Image.Image):
    # ensure it's RGB
    img = img_obj.convert("RGB")

    # resize
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

    # convert to np array (float32)
    img = np.array(img).astype("float32")

    # convert MEAN/STD to float32 as well
    mean = MEAN.astype("float32")
    std = STD.astype("float32")

    # rescale
    img = img * np.float32(RESCALE)

    # normalize
    img = (img - mean) / std

    # convert to channel-first
    img = np.transpose(img, (2, 0, 1))

    # add batch dimension
    img = np.expand_dims(img, axis=0)

    # ensure final dtype is float32
    return img.astype("float32")


# -----------------------------------------------------
# LOAD ONNX MODEL
# -----------------------------------------------------
try:
    session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
except ort.OnnxRuntimeError as e:
    print(f"Error loading ONNX model: {e}")
    exit()

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# -----------------------------------------------------
# PREDICT FUNCTION
# Modified to accept a PIL Image object.
# -----------------------------------------------------
def predict(img_obj: Image.Image):
    inputs = preprocess_image(img_obj)
    outputs = session.run([output_name], {input_name: inputs})
    logits = outputs[0]

    pred_id = int(np.argmax(logits))
    return pred_id, id2label[pred_id]

# -----------------------------------------------------
# FLASK ROUTE TO HANDLE IMAGE UPLOAD
# -----------------------------------------------------
@app.route("/predict", methods=["POST"])
def image_prediction():
    # 1. Check if the 'file' is in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    # 2. Check if a filename is provided (i.e., a file was selected)
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            # 3. Read the image data from the file stream directly into a PIL Image
            image_data = file.read()
            image = Image.open(BytesIO(image_data))
            
            # 4. Run the prediction
            pred_id, label = predict(image)

            # 5. Format and return the required response
            response = {
                "Class ID": pred_id,
                "Predicted Label": label
            }
            
            # For the required output format: Class ID: 4\nPredicted Label: Potato___Early_Blight
            # We will return the structured JSON and let the client format it,
            # but if you strictly want the raw text:
            # return f"Class ID: {pred_id}\nPredicted Label: {label}", 200
            
            return jsonify(response), 200

        except Exception as e:
            # Catch any issues during processing (e.g., corrupted file)
            return jsonify({"error": f"Internal server error during prediction: {str(e)}"}), 500
    
    # Should not be reached, but as a fallback
    return jsonify({"error": "Unknown error processing file"}), 500

# -----------------------------------------------------
# RUN SERVER
# -----------------------------------------------------
if __name__ == "__main__":
    # Remove the old test prediction
    # image_path = "b2600118-800px-wm.jpg"
    # pred_id, label = predict(image_path)
    # print("Class ID:", pred_id)
    # print("Predicted Label:", label)

    # Run the Flask development server
    print("Starting Flask server on accessible interfaces on port 5000/predict")
    print("Send a POST request with an image file named 'file' to this endpoint.")
    
    # --- CHANGE THIS LINE ---
    # app.run(host='127.0.0.1', port=5000)
    
    # --- TO THIS LINE ---
    app.run(host='0.0.0.0', port=5000)