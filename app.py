from flask import Flask, request, send_from_directory, Response, jsonify
from PIL import Image, ImageDraw
import torch
import cv2
from apply_nms import apply_nms
import numpy as np
import os
import io
import uuid
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import base64

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

UPLOAD_FOLDER = "uploads"

app = Flask(__name__)
app.debug = True
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]

    if file.filename == "":
        return "No selected file", 400

    # Save the uploaded image
    filename = os.path.join(app.config["UPLOAD_FOLDER"], str(uuid.uuid4()) + ".jpg")
    file.save(filename)

    # Process the image with cv2
    image = Image.open(filename)

    texts = [["a photo of tissue box", "a photo of a mouse", "a photo of helmet", "a photo of a calculator", "a photo of a book", "a photo of a bottle", "a photo of a binoculars", "a photo of a water bottle", "a photo of a remote control", "a photo of a table lamp", "a photo of a tape dispenser cutter", "a photo of a playing card", "a photo of thing", "a photo of any object", "a photo of a tool", "a photo of a object", "goods", "a photo of a household", "thing", "a photo of a pencil", "a photo of a ball", "a photo of a toy", "a photo of a round object", "not background", "a photo of hard object", "a photo of small object", "a photo of playing tool", "a photo of tennis ball", "a photo of ballpen", "a photo of white pen", "a photo of blue water gun", "a photo of gun", "a photo of blue object", "a photo of a white object", "a photo of toy", "a photo of a square object"]]
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    # Print detected objects and rescaled box coordinates
    score_threshold = 0.1
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        if score >= score_threshold:
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")


    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Color and width for the bounding box
    box_color = "red"
    box_width = 4

    score_threshold = 0.1
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []

    for box, score, label in zip(boxes, scores, labels):
        if score >= score_threshold:
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filtered_labels.append(label)

    # Apply NMS
    indices_to_keep = apply_nms(filtered_boxes, filtered_scores, threshold=0.5)

    detected_objects = []

    for idx in indices_to_keep:
        box = filtered_boxes[idx]
        score = filtered_scores[idx]
        label = filtered_labels[idx]

        # Draw the bounding box
        top_left = (box[0], box[1])
        bottom_right = (box[2], box[3])
        draw.rectangle([top_left, bottom_right], outline=box_color, width=box_width)

        # Append object information to the detected_objects list
        detected_objects.append({
            "object": f"{text[label]}",
            "confidence": f"{round(score.item(), 3)}",
            "location": {
                "top_left": {
                    "x": f"{box[0]}", 
                    "y": f"{box[1]}"
                },
                "bottom_right": {
                    "x": f"{box[2]}", 
                    "y": f"{box[3]}"
                }
            }
        })


    
    os.remove(filename)

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_byte_array = buffered.getvalue()

    # Encode image to base64
    base64_encoded_img = base64.b64encode(img_byte_array).decode('utf-8')

    response = {
        "detected_objects": detected_objects,
        "image": f"data:image/jpeg;base64,{base64_encoded_img}"
    }

    return jsonify(response)



if __name__ == "__main__":
    app.run(debug=True)
