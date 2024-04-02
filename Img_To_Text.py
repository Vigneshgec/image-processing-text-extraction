from google.colab.patches import cv2_imshow
import numpy as np
import easyocr
import requests

API_KEY = "Your_API_KEY"
def load_east_model(east_model_path):
    """
    Load the EAST model for text detection.
    """
    return cv2.dnn.readNet(east_model_path)

def detect_text_regions(image_path, east_net, conf_threshold=0.5, nms_threshold=0.4):
    """
    Detect text regions in an image using the EAST model.
    """
    image = cv2.imread(image_path)
    orig = image.copy()
    (H, W) = image.shape[:2]

    # Set the new width and height and then determine the ratio in change
    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    east_net.setInput(blob)
    (scores, geometry) = east_net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    # decode the predictions
    (rects, confidences) = decode_predictions(scores, geometry, conf_threshold)
    boxes = non_max_suppression(np.array(rects), probs=confidences, overlapThresh=nms_threshold)

    # scale the bounding boxes back to the original image size
    boxes = [(int(startX * rW), int(startY * rH), int(endX * rW), int(endY * rH)) for (startX, startY, endX, endY) in boxes]
    return boxes, orig

def decode_predictions(scores, geometry, conf_threshold=0.5):
    """
    Decode the predictions from the EAST detector.
    """
    # Initialize our list of bounding box rectangles and corresponding confidence scores
    rects = []
    confidences = []

    # loop over the number of rows in the scores volume
    for y in range(0, scores.shape[2]):
        # extract the scores (probabilities) and geometrical data to derive potential bounding box coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns in the score row
        for x in range(0, scores.shape[3]):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < conf_threshold:
                continue

            # compute the offset factor as our resulting feature maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    """
    Perform non-maximum suppression to suppress weak, overlapping bounding boxes.
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index value to the list of picked indexes, then initialize the suppression list (i.e., indexes that will be deleted) using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # delete all indexes from the index list that are in the suppression list
        idxs = np.delete(idxs, suppress)

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int")

def refine_text_with_gemini(text):
    """
    Refines the given text using Google's Gemini API, aiming to improve the
    clarity and correctness of the sentences.
    """
    # Adjusting the prompt to focus on corrections
    prompt = f"Correct any errors in the following text: {text}"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]
    }

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={API_KEY}",
        json=payload
    )

    if response.status_code == 200:
        # Parsing the response based on the structure you provided
        try:
            refined_parts = response.json()['candidates'][0]['content']['parts']
            refined_text = " ".join([part['text'] for part in refined_parts])
        except (KeyError, IndexError):
            refined_text = "Error parsing response."
        return refined_text
    else:
        print("Error:", response.status_code)
        print(response.text)  # To understand more about the error
        return text  # Return the original text if API call fails

def main(image_path, east_model_path):
    east_net = load_east_model(east_model_path)
    boxes, orig_image = detect_text_regions(image_path, east_net)

    reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader
    all_recognized_text=[]
    all_refined_text=[]
    for (startX, startY, endX, endY) in boxes:
        # Correct any out-of-bound coordinates
        startX, startY = max(0, startX), max(0, startY)
        endX, endY = min(orig_image.shape[1] - 1, endX), min(orig_image.shape[0] - 1, endY)

        # Ensure valid ROI dimensions
        if startX >= endX or startY >= endY:
            print(f"Skipped due to invalid dimensions: startX={startX}, startY={startY}, endX={endX}, endY={endY}")
            continue

        # Extract the actual padded ROI
        roi = orig_image[startY:endY, startX:endX]

        # Use EasyOCR to recognize text in the ROI
        try:
            ocr_result = reader.readtext(roi, paragraph=True)
            for result in ocr_result:
                if len(result) == 2:  # If the result structure is different
                    bbox, text = result
                    prob = None  # Probability is not available
                else:
                    bbox, text, prob = result
                all_recognized_text.append(text)
    #print(f"Recognized Text: {text}")
                # Optionally refine text using an external API
                refined_text = refine_text_with_gemini(text)
                all_refined_text.append(refined_text)
    #print(f"Refined Text: {refined_text}")
        except Exception as e:
            print(f"Error processing ROI: {e}")
            continue

        # Draw rectangles around detected text regions
        cv2.rectangle(orig_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Display the image with detected text regions highlighted
    cv2_imshow(orig_image)

    print("Recognized text is:",all_recognized_text)
    print("refined text is:",all_refined_text)

if _name_ == "_main_":
    image_path = r"/content/sample_image.png"
    east_model_path = r"/content/drive/MyDrive/Colab_Notebooks/frozen_east_text_detection.pb"
    main(image_path, east_model_path)