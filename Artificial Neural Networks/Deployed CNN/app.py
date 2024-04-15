import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_file
from PIL import Image
import uuid
import base64
import io


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# function to classify the image
def classify_digit(model, image):
    img = np.array(image)
    
    # Convert image to grayscale if it has more than one channel
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    img = cv2.resize(img, (28, 28))  # Resize to 28x28 pixels
    img = np.invert(img)  # Invert the image (if necessary)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension

    prediction = model.predict(img)
    confidence = round(np.max(prediction) * 100, 2)

    print(f"This is the confidence: {confidence}")
    
    # Resize the image back to its original size
    resized_image = cv2.resize(np.squeeze(img), (200, 200))
    resized_image = Image.fromarray(resized_image)

    return prediction, confidence, resized_image



# function to resize the image
def resize_image(image, target_size):
    img = Image.open(image)
    
    # Convert the image to RGB mode if it's not in that format
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Check if the image dimensions are larger than the target size
    if img.width > target_size[0] or img.height > target_size[1]:
        img.thumbnail(target_size)
    
    return img

# Get the directory path of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join("static", "uploads")
THUMBNAIL_FOLDER = os.path.join("static", "uploads", "thumbnails")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["THUMBNAIL_FOLDER"] = THUMBNAIL_FOLDER

# Create the uploads and thumbnails directories if they don't exist
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

if not os.path.exists(app.config["THUMBNAIL_FOLDER"]):
    os.makedirs(app.config["THUMBNAIL_FOLDER"])

# route for home page
@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    confidence = None
    uploaded_image = None
    thumbnail_image = None
    img_str = None
    
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Generate a unique filename
            unique_filename = str(uuid.uuid4()) + ".png"
            temp_image_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
            uploaded_file.save(temp_image_path)
            
            # Construct the full path to the model file
            model_path = os.path.join(SCRIPT_DIR, "handwrittendigit.h5")
            model = tf.keras.models.load_model(model_path)
            
            # Resize the image if necessary and create a thumbnail
            resized_image = resize_image(temp_image_path, (200, 200))
            thumbnail_filename = "thumbnails/" + unique_filename
            thumbnail_path = os.path.join(app.config["UPLOAD_FOLDER"], thumbnail_filename)
            resized_image.save(thumbnail_path)
            
            prediction, confidence, resized_image = classify_digit(model, resized_image)
            
            # Using np.argmax(prediction) will reveal the number with the highest probability
            result = f'The digit is probably a {np.argmax(prediction)}, Confidence: {confidence:.2f}%'
            

            # Encode the image to base64
            buffered = io.BytesIO()
            resized_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
    return render_template('index.html', result=result, confidence=confidence, image=img_str)


# route to serve uploaded image
@app.route('/uploads/<filename>')
def send_uploaded_image(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True)
