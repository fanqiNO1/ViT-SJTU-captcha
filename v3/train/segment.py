import os

import cv2
import numpy as np
from sklearn.svm import SVC
from tqdm import tqdm


def segment(image: np.ndarray) -> list[np.ndarray]:
    KUN_RATIO = 0.268  # Don't ask.
    KUN_ASPECT_RATIO = 1.77
    # Convert image to grayscale and apply threshold
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Revert italic by affine transformation
    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2),
                                -0.22, 1)
    gray = cv2.warpAffine(gray, M, (image.shape[1], image.shape[0]))
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)

    # Find bounding box for the whole cropped region
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = 0, 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    # Crop the image
    cropped_to_w = max_x - min_x

    # Re-find contours on the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    bboxes = []
    for contour in contours:
        this_x, this_y, this_w, this_h = cv2.boundingRect(contour)

        if (cropped_to_w * KUN_RATIO < this_w
                or this_w / this_h >= KUN_ASPECT_RATIO):
            # Apply SVM classification to split the wide bounding box
            split_points = np.column_stack(
                np.where(thresh[this_y:this_y + this_h,
                                this_x:this_x + this_w] > 0))
            if len(split_points) > 1:
                # Train the SVM model with two classes based on x-coordinates
                model = SVC(kernel='linear', C=1.0, probability=True)
                labels_init = (split_points[:, 1] > np.median(
                    split_points[:, 1])).astype(int)
                model.fit(split_points, labels_init)

                # Predict the labels
                labels = model.predict(split_points)

                # Find the mean x-values of both clusters
                mean_x_cluster_0 = np.mean(split_points[labels == 0, 1])
                # mean_y_cluster_0 = np.mean(split_points[labels == 0, 0])
                mean_x_cluster_1 = np.mean(split_points[labels == 1, 1])
                # mean_y_cluster_1 = np.mean(split_points[labels == 1, 0])
                split_x = int((mean_x_cluster_0 + mean_x_cluster_1) / 2)

                # Ensure a significant separation before splitting
                if abs(mean_x_cluster_0 - mean_x_cluster_1) > this_w * 0.4:
                    # Add two separate bounding boxes
                    bboxes.append((this_x, this_y, split_x, this_h))
                    bboxes.append(
                        (this_x + split_x, this_y, this_w - split_x, this_h))
                else:
                    bboxes.append((this_x, this_y, this_w, this_h))
            else:
                bboxes.append((this_x, this_y, this_w, this_h))
        else:
            bboxes.append((this_x, this_y, this_w, this_h))

    # Sort the bounding boxes by x-coordinate
    bboxes = sorted(bboxes, key=lambda x: x[0])

    # Patch i and j
    # Re-iterate the bboxes, find width < 10 and height < 10 bboxes
    # and merge it with the closest bbox other than itself
    for i in range(len(bboxes)):
        x, y, w, h = bboxes[i]
        if w < 9 and h < 9:
            # Find the closest bbox other than itself
            closest_bbox = None
            closest_distance = float('inf')
            for j in range(len(bboxes)):
                if i == j or bboxes[j] is None:
                    continue
                x2, y2, w2, h2 = bboxes[j]
                distance = abs(x - x2)
                if distance < closest_distance:
                    closest_bbox = j
                    closest_distance = distance
            # Merge the two bboxes
            x2, y2, w2, h2 = bboxes[closest_bbox]
            bboxes[closest_bbox] = (
                min(x, x2),
                min(y, y2),
                max(x + w, x2 + w2) - min(x, x2),
                max(y + h, y2 + h2) - min(y, y2),
            )
            bboxes[i] = None

    # Post process the bboxes
    bboxes = [bbox for bbox in bboxes if bbox is not None]

    # Sort the bounding boxes by x-coordinate again.
    # If 6 bboxes are found, merge two closest bboxes.
    bboxes = sorted(bboxes, key=lambda x: x[0])

    while len(bboxes) >= 6:
        min_distance = float('inf')
        merge_index = None
        for i in range(5):
            x, y, w, h = bboxes[i]
            x2, y2, w2, h2 = bboxes[i + 1]
            distance = abs(x + w - x2)
            if distance < min_distance:
                min_distance = distance
                merge_index = i
        if min_distance < 5:
            x, y, w, h = bboxes[merge_index]
            x2, y2, w2, h2 = bboxes[merge_index + 1]
            bboxes[merge_index] = (x, y, w + w2, h)
            bboxes[merge_index + 1] = None
        bboxes = [bbox for bbox in bboxes if bbox is not None]

    # Final assertion
    bboxes = sorted(bboxes, key=lambda x: x[0])

    # Extract segmented character images
    segmented_images = []
    for x, y, w, h in bboxes:
        this_char = thresh[y:y + h, x:x + w]
        segmented_images.append(this_char)

    return segmented_images


def main():
    # Get the files
    files = os.listdir('labelled')
    # Create the output directory
    if not os.path.exists('data'):
        os.makedirs('data')
    # Create the counter with key as label and value as count
    counter = {}
    # Iterate over the files
    for file in tqdm(files):
        path = os.path.join('labelled', file)
        # Get the image
        image = cv2.imread(path)

        # Extract the labels from the filename
        labels = list(file.split('_')[0])

        # Call the updated segment function
        segmented_images = segment(image)

        # Check if the number of segmented images matches the number of labels
        if len(segmented_images) != len(labels):
            print(f'Warning: Mismatch in file {file}. '
                  f'Found {len(segmented_images)} segments '
                  f'but expected {len(labels)} characters.')
            # Skip this file or implement a fallback strategy
            continue

        # Iterate over the segmented images and corresponding labels
        for i, (image, label) in enumerate(zip(segmented_images, labels)):
            # Resize the image to 28x28 if it's not already
            image = cv2.resize(image, (28, 28))

            # Get the counter
            count = counter.get(label, 0)
            counter[label] = count + 1

            # Save the image
            cv2.imwrite(os.path.join('data', f'{label}_{count}.png'), image)

    # Print the counter
    for label, count in counter.items():
        print(f'There are {count} images of label {label}.')


if __name__ == '__main__':
    main()
