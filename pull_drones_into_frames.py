import os
import cv2
import numpy as np

def put_sample_in_frame(frame, sample):
    # Place the sample at a random spot in the image, with a random flip in both axes
    y = np.random.randint(0, frame.shape[0] - max(sample.shape))
    x = np.random.randint(0, frame.shape[1] - max(sample.shape))
    print(f'0 < {x} (x) < {frame.shape[1] - max(sample.shape)} (calculated max width)')
    print(f'0 < {y} (y) < {frame.shape[0] - max(sample.shape)} (calculated max height)')
    flip = np.random.randint(0, 2)
    if flip:
        sample = np.flip(sample, axis=0)
    flip = np.random.randint(0, 2)
    if flip:
        sample = np.flip(sample, axis=1)

    # Finally, apply a bounded random rotation
    rotation = np.random.randint(0, 4)
    sample = np.rot90(sample, rotation)

    print(f'y: {y}, x: {x}, w: {sample.shape[1]}, h: {sample.shape[0]}')
    print(f'y: {y / frame.shape[1]: .3f}, x: {x / frame.shape[1]: .3f}, '
          f'w: {sample.shape[1] / frame.shape[1]: .3f}, h: {sample.shape[0] / frame.shape[0]: .3f}')
    frame[y:y+sample.shape[0], x:x+sample.shape[1], :] = sample

    # # Apply a random brightness and contrast to the whole image
    # brightness = np.random.randint(-50, 50)
    # contrast = np.random.randint(-50, 50)
    # image = np.clip(image * (1 + contrast / 100) + brightness, 0, 10)
    return frame, (x, y, sample.shape[1], sample.shape[0])


if __name__ == '__main__':
    sample_img = np.zeros((50, 50, 3))
    sample_img[:, sample_img.shape[1] // 2:, :] = 255
    image = cv2.imread('./AppMAIS11s_labeled_data/images/video_AppMAIS11L@2023-05-02@15-45-00_frame_175.png')
    x = 0
    y = 0
    x2 = image.shape[1]
    y2 = image.shape[0] - 150
    cropped_image = image[y:y2, x:x2, :]
    cropped_image_with_sample = put_sample_in_frame(cropped_image, sample_img)
    image[y:y2, x:x2, :] = cropped_image_with_sample
    # Draw rectangle around region of interest
    cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('', image)
    cv2.waitKey(0)