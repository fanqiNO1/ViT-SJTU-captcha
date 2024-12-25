import os

import cv2
from tqdm import tqdm


def segement(image, label):
    # Split the label
    labels = list(label.split("_")[0])
    # Grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Remove the noise
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # Threshold the image
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # Charater Segmentation using contours
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort the contours
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    # Create the bounding boxes
    bounding_boxes = []
    # Check if "i" or "j" is present
    label_i = 0
    contour_i = 0
    length_diff = len(contours) - len(labels)
    while label_i < len(labels) and contour_i < len(contours):
        if (labels[label_i] == "i" or labels[label_i] == "j") and length_diff > 0:
            # Get the bounding box
            x1, y1, w1, h1 = cv2.boundingRect(contours[contour_i])
            x2, y2, w2, h2 = cv2.boundingRect(contours[contour_i + 1])
            # Create the bounding box
            x, y = min(x1, x2), min(y1, y2)
            w, h = max(x1 + w1, x2 + w2) - x, max(y1 + h1, y2 + h2) - y
            # Append the bounding box
            bounding_boxes.append([x, y, w, h])
            # Increment the contour index
            contour_i += 2
            # Decrement the length difference
            length_diff -= 1
        else:
            # Get the bounding box
            x, y, w, h = cv2.boundingRect(contours[contour_i])
            # Append the bounding box
            bounding_boxes.append([x, y, w, h])
            # Increment the contour index
            contour_i += 1
        # Increment the label index
        label_i += 1
    # Get the images
    images = []
    for i, box in enumerate(bounding_boxes):
        x, y, w, h = box
        # Get the image
        image_i = image[y:y + h, x:x + w]
        # Resize the image
        image_i = cv2.resize(image_i, (28, 28))
        # Append the image
        images.append({
            "image": image_i,
            "label": labels[i]
        })
    # Return the images
    return images


def main():
    # Get the files
    files = os.listdir("labelled")
    # Create the output directory
    if not os.path.exists("data"):
        os.makedirs("data")
    # Create the counter with key as label and value as count
    counter = {}
    # Iterate over the files
    for file in tqdm(files):
        path = os.path.join("labelled", file)
        # Get the image
        image = cv2.imread(path)
        # Resize the image
        image = cv2.resize(image, dsize=(0, 0), fx=28, fy=28)
        # Segement the image
        images = segement(image, file)
        # Iterate over the images
        for data in images:
            image, label = data["image"], data["label"]
            # Get the counter
            count = counter.get(label, 0)
            counter[label] = count + 1
            # Save the image
            cv2.imwrite(os.path.join("data", f"{label}_{count}.png"), image)
            # Show the image
            # cv2.imshow(f"{label}_{count}", image)
            # cv2.waitKey(0)
    # Print the counter
    for label, count in counter.items():
        print(f"There are {count} images of label {label}.")


if __name__ == "__main__":
    main()
