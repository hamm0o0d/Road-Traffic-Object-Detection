import streamlit as sl
import os
import matplotlib.pyplot as plt
import tempfile
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from PIL import Image
import cv2

themodel = YOLO('best wieghts_100 epoch.pt')


def object_detection(image_data):
    # Create a temporary file to save the uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
        temp_image.write(image_data)
        temp_image_path = temp_image.name

    # Assuming your model's `predict` method takes a file path as input
    theresults = themodel.predict(source=temp_image_path, conf=0.25)

    image = Image.open(temp_image_path)

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Define the detected objects with their bounding box coordinates, class labels, and confidences
    # Each object is represented as (x, y, width, height, label, confidence)
    # Replace these with the actual information about detected objects

    # Draw bounding boxes and labels on the image
    for xyxy, conf, cls in zip(theresults[0].boxes.xyxy.cpu().numpy(), theresults[0].boxes.conf.cpu().numpy(),
                               theresults[0].boxes.cls.cpu().numpy()):
        x1, y1, x2, y2 = xyxy
        confidence = conf
        class_id = theresults[0].names[cls]

        # Use the default system font
        label_font = ImageFont.load_default()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
        draw.text((x1, y1 - 15), f"{class_id} ({confidence:.2f})", fill="red", font=label_font)

    # Display the image with bounding boxes and labels using Matplotlib
    plt.imshow(image)
    plt.axis('off')  # Turn off axis labels

    plt.savefig(f'C:/Users/ahmed/PycharmProjects/Object Detection/image.jpg', bbox_inches='tight', pad_inches=0.1)

    # Show the image
    plt.show()

    # Remove the temporary file
    os.remove(temp_image_path)

    return


sl.title("Road Traffic Object Detection Project using YOLOv8")
sl.text("team members : Ahmed Hany, Ahmed Esmat and Ismail Sherif")
# Add a file uploader widget
uploaded_image = sl.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
image_data=None
if uploaded_image is not None:
    image_data = uploaded_image.read()  # Read the image data as bytes
    # Now, you can use image_data with libraries like OpenCV, Pillow, etc.
    object_detection(image_data)
    with open('image.jpg', 'rb') as image_file:
        my_image = image_file.read()

    sl.image(my_image, caption="Uploaded Image", use_column_width=True)

