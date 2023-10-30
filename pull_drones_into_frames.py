import os
import cv2
import numpy as np
from typing import Tuple


def put_sample_in_frame(image, sample):
    # Place the sample at a random spot in the image, with a random flip in both axes
    y = np.random.randint(0, image.shape[0] - max(sample.shape))
    x = np.random.randint(0, image.shape[1] - max(sample.shape))
    # print(f'0 < {x} (x) < {image.shape[1] - max(sample.shape)} (calculated max width)')
    # print(f'0 < {y} (y) < {image.shape[0] - max(sample.shape)} (calculated max height)')
    flip = np.random.randint(0, 2)
    if flip:
        sample = np.flip(sample, axis=0)
    flip = np.random.randint(0, 2)
    if flip:
        sample = np.flip(sample, axis=1)

    # Finally, apply a bounded random rotation
    rotation = np.random.randint(0, 4)
    sample = np.rot90(sample, rotation)

    # print(f'x: {x}, y: {y}, w: {sample.shape[1]}, h: {sample.shape[0]}')
    # print(f'y: {y / image.shape[1]: .3f}, x: {x / image.shape[1]: .3f}, '
    #       f'w: {sample.shape[1] / image.shape[1]: .3f}, h: {sample.shape[0] / image.shape[0]: .3f}')
    image[y:y + sample.shape[0], x:x + sample.shape[1], :] = sample

    # # Apply a random brightness and contrast to the whole image
    # brightness = np.random.randint(-50, 50)
    # contrast = np.random.randint(-50, 50)
    # image = np.clip(image * (1 + contrast / 100) + brightness, 0, 10)
    # Output the image and a tuple of the bounding box coordinates (center_x, center_y, width, height), all relative
    # to the image size
    # return image, (x / image.shape[1], y / image.shape[0],
    #                sample.shape[1] / image.shape[1], sample.shape[0] / image.shape[0])
    rel_center_x, rel_center_y, rel_width, rel_height = convert_actual_xywh_to_relative(
        image.shape, x, y, sample.shape[1], sample.shape[0]
    )
    return image, (rel_center_x, rel_center_y, rel_width, rel_height)


def draw_rectangle_around_sample(image, x, y, w, h, color=(0, 0, 255)):
    # x y w and h should all be given as a fraction of the image size.
    # x and y denote the center of the rectangle
    x, y, w, h = convert_relative_xywh_to_actual(image.shape, x, y, w, h)

    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    return image

def convert_actual_xywh_to_relative(image_shape, x, y, w, h):
    # x y denote the top left corner and w and h are the width and height in pixels
    w = w / image_shape[1]
    h = h / image_shape[0]
    # x and y will be converted to the center of the rectangle
    x = x / image_shape[1] + (w / 2)
    y = y / image_shape[0] + (h / 2)
    return x, y, w, h

def convert_relative_xywh_to_actual(image_shape, x, y, w, h):
    # x y w and h should all be given as a fraction of the image size.
    # x and y denote the center of the rectangle
    w = int(w * image_shape[1])
    h = int(h * image_shape[0])
    x = int(x * image_shape[1] - (w / 2))
    y = int(y * image_shape[0] - (h / 2))

    return x, y, w, h

def convert_xywh_for_larger_image(image_shape_before, image_shape_after, x, y, w, h):
    # x y w and h should all be given as a fraction of the image size.
    # x and y denote the center of the rectangle
    w = int(w * image_shape_before[1])
    h = int(h * image_shape_before[0])
    x = int(x * image_shape_before[1])
    y = int(y * image_shape_before[0])

    x = x / image_shape_after[1]
    y = y / image_shape_after[0]
    w = w / image_shape_after[1]
    h = h / image_shape_after[0]

    return x, y, w, h

def demo_with_object(image_path: str, object_width: int, object_height: int, roi: Tuple[int, int, int, int]):
    sample_obj = np.zeros((object_height, object_width, 3))
    sample_obj[:, sample_obj.shape[1] // 2:, :] = 255
    image = cv2.imread(image_path)
    roi_x, roi_y, roi_x2, roi_y2 = roi
    cropped_image = image[roi_y:roi_y2, roi_x:roi_x2, :]
    cropped_image_with_sample, (drone_x, drone_y, drone_w, drone_h) = put_sample_in_frame(cropped_image, sample_obj)
    # cropped_image_with_sample = draw_rectangle_around_sample(cropped_image_with_sample, drone_x, drone_y, drone_w, drone_h)
    image[roi_y:roi_y2, roi_x:roi_x2, :] = cropped_image_with_sample
    drone_x, drone_y, drone_w, drone_h = convert_xywh_for_larger_image(
        cropped_image.shape, image.shape, drone_x, drone_y, drone_w, drone_h
    )
    print(f'Inserted drone is at location x: {drone_x}, y: {drone_y}, w: {drone_w}, h: {drone_h}')
    image = draw_rectangle_around_sample(image, drone_x, drone_y, drone_w, drone_h)
    # Draw rectangle around region of interest
    cv2.rectangle(image, (roi_x, roi_y), (roi_x2, roi_y2), (0, 255, 0), 2)
    return image

def pull_drone_samples_from_images(images_and_labels):
    # For each image, get the indices of the labels_list that are drones
    # Then, for each drone, get the image and label, and put the image in the frame
    # Then, save the image and label to a new folder
    pass

def load_labels_from_path(path):
    labels = []
    label_filenames = os.listdir(path)
    for label_filename in label_filenames:
        with open(f"{path}/{label_filename}", "r") as f:
            lines = f.readlines()
            image_label = []
            for line in lines:
                bee_label = line.split(" ")
                bee_label = [float(x) for x in bee_label]
                image_label.append(bee_label)

            labels.append(image_label)
    return labels

if __name__ == '__main__':
    #### We have discussed this and it may not be the best approach as it can teach the ultralytics_model about shifting HSV values
    #### for the background
    # Demo how the function works
    sample_height = 100
    sample_width = 50

    frame_path = os.path.abspath('./AppMAIS11s_labeled_data/images/video_AppMAIS11L@2023-05-02@15-45-00_frame_175.png')
    roi = (0, 0, 640, 480 - 150)
    frame = demo_with_object(frame_path, sample_width, sample_height, roi)
    cv2.imshow('', frame)
    cv2.waitKey(0)

    # Then load in the images and labels_list and apply the function to each image with any example of drones (label 0)
    # and save the results to a new folder
    images_path = ''
    labels_path = ''
    images = [cv2.imread(os.path.join(images_path,image_path)) for image_path in os.listdir(images_path)]

    # Get the labels_list, and then get the indices of the labels_list that are drones
    labels = load_labels_from_path(labels_path)
    drone_labels_in_images = []
    for image, label in zip(images, labels):
        for bee_label in label:
            if bee_label[0] == 0:
                drone_labels_in_images.append((image, bee_label))




