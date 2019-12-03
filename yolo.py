import cv2
import numpy as np


def get_bounding_images_p(img_path, confidence=0.6, threshold=0.3):
    img = cv2.imread(img_path)
    return get_bounding_images(img, confidence, threshold)


def get_bounding_images(img, confidence=0.6, threshold=0.3):
    yolo_net = cv2.dnn.readNetFromDarknet("yolo/yolov3.cfg", "yolo/yolov3.weights")

    layer_names = yolo_net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]

    height, width, channels = img.shape

    # Convert image to blob
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            detected_confidence = scores[class_id]

            # Filter predictions based on confidence
            if detected_confidence > confidence:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Top left of box
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(detected_confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    cropped_images = []

    if len(indexes) > 0:
        for i in indexes.flatten():
            # Only look at cars
            if class_ids[i] == 2:
                x, y, w, h = boxes[i]

                # Crop image around box
                cropped_img = img[y: y + h, x: x + w]
                cropped_images.append((cropped_img, boxes[i]))

    return img, cropped_images
