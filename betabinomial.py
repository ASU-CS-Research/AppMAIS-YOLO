import matplotlib.pyplot as plt
from scipy.stats import betabinom, binom, beta
import numpy as np
from typing import List, Optional, Tuple
import ultralytics
import os
import cv2 as cv


def get_model_results(model, images, labels=None, image_filenames=None):
    results = model.predict(images, conf=0.64)
    images_for_model_output = [np.copy(image) for image in images]
    images_for_label_output = [np.copy(image) for image in images]
    # Apply bounding boxes to images
    for i, (model_image, label_image, image_filename, result, label) in enumerate(zip(
            images_for_model_output, images_for_label_output, image_filenames, results, labels
        )):
        # Plot the model output
        bounding_boxes = result.boxes
        for box in bounding_boxes:
            x1, y1, x2, y2, conf, class_id = box.data.tolist()[0]
            cv.rectangle(model_image, (int(x1), int(y1)), (int(x2), int(y2)),
                         color=(255 * abs(class_id - 1), 0, 255 * class_id), thickness=2)
            cv.putText(model_image, f'{conf * 100: .2f}', (int(x1), int(y1)), cv.FONT_HERSHEY_SIMPLEX, 1,
                       (255 * abs(class_id - 1), 0, 255 * class_id), 2)
        # Plot the human made labels_list
        for box in label:
            # box is given as class_id, center_x, center_y, w, h all relative to image width and height.
            # We need to convert to x1, y1, x2, y2
            class_id, center_x, center_y, w, h = box
            x1 = int((center_x - w / 2) * label_image.shape[1])
            y1 = int((center_y - h / 2) * label_image.shape[0])
            x2 = int((center_x + w / 2) * label_image.shape[1])
            y2 = int((center_y + h / 2) * label_image.shape[0])
            cv.rectangle(label_image, (x1, y1), (x2, y2), (255 * abs(class_id - 1), 0, 255 * class_id), 2)
        images_for_label_output[i] = label_image
        images_for_model_output[i] = model_image

    return results, images_for_model_output, images_for_label_output


def beta_binom_on_data(model, images: List[np.ndarray], labels: List[List[List[float]]], image_filenames: List[str],
                       copy_images_and_labels: Optional[bool] = False, output_dir: Optional[str] = None,
                       verbose: Optional[bool] = False) -> List[float]:
    results, model_images, label_images = get_model_results(
        model, images, labels, image_filenames
    )
    log_likelihoods = []
    if copy_images_and_labels and output_dir is not None:
        for model_image, label_image, image_filename in zip(model_images, label_images, image_filenames):
            # save the images
            os.makedirs(os.path.join(output_dir, 'model_outputs'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'label_outputs'), exist_ok=True)
            model_output_path = os.path.join(output_dir, 'model_outputs', image_filename.replace('.png', '') + ".png")
            cv.imwrite(model_output_path, model_image)
            label_output_path = os.path.join(output_dir, 'label_outputs', image_filename.replace('.png', '') + ".png")
            cv.imwrite(label_output_path, label_image)

    for result, label, filename in zip(results, labels, image_filenames):
        bounding_boxes = result.boxes
        # the prior distribution of alpha and beta should not be zero because that would imply that either category does
        # not exist and therefore n should be at least 2 and alpha and beta should be at least 1
        n = 2
        a = 1
        b = 1
        n_true = 2
        a_true = 1
        b_true = 1
        for box in bounding_boxes:
            n += 1
            x1, y1, x2, y2, conf, class_id = box.data.tolist()[0]
            a += 1 if class_id == 1 else 0
            b += 1 if class_id == 0 else 0

        p = float(a/n) if n != 0 else 0

        for box in label:
            if len(box) == 0:
                continue

            n_true += 1
            a_true += 1 if box[0] == 1 else 0
            b_true += 1 if box[0] == 0 else 0

        # n_true = 10
        # a_true = 7
        # b_true = n_true - a_true
        p_true = a_true/n_true if n_true != 0 else 0
        n_scaled = n_true * n
        a_scaled = p * n_scaled
        b_scaled = n_scaled - a_scaled


        #mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')

        # mean, var, skew, kurt = betabinom.stats(n_scaled, a_scaled, b_scaled, moments='mvsk')

        # print("mean: ", mean)
        # print("var: ", var)
        # print("skew: ", skew)
        # print("kurt: ", kurt)
        # print("p: ", p)

        # plot the distribution
        # x = np.arange(binom.ppf(0.01, n_scaled, p), binom.ppf(0.99, n_scaled, p))
        #
        # plt.plot(x, betabinom.pmf(x, n_scaled, a_scaled, b_scaled), 'bo', ms=8, label='betabinom pmf')
        #plt.show()
        #do the same for a binomial distribution
        # plt.plot(x, binom.pmf(x, n_scaled, p), 'ro', ms=8, label='binom pmf')
        # plt.legend(loc='best')
        #
        # # make a vertical line at the true value of p
        # plt.axvline(x= p_true * n_scaled, color='k', linestyle='--')

        x = n_scaled * p_true
        beta_binom_prob = betabinom.pmf(x, n_scaled, a_scaled, b_scaled)

        if verbose:
            print("Image filename: ", filename)
            print("n_predicted: ", n)
            print("a_predicted: ", a)
            print("b_predicted: ", b)
            print("p_predicted: ", p)
            print("n_true: ", n_true)
            print("a_true: ", a_true)
            print("b_true: ", b_true)
            print("p_true: ", p_true)
            # binom_prob = binom.pmf(x, n_scaled, p)
            print("beta_binom_prob: ", beta_binom_prob)
            # print("binom_prob: ", binom_prob)
            # plt.show()

            print("log likelihood beta binom: ", np.log(beta_binom_prob))
            print()
            # print("log likelihood binom: ", np.log(binom_prob))

    return log_likelyhoods

def rmse_on_data(model, images: List[np.ndarray], labels: np.ndarray, image_filenames: List[str], copy_images_and_labels: Optional[bool] = False, output_dir: Optional[str] = None):
    results = get_model_results(model, images, labels, image_filenames, copy_images_and_labels, output_dir)

    drones_error = 0
    workers_error = 0

    for result, label, filename in zip(results, labels, image_filenames):
        drones = 0
        workers = 0
        drones_hat = 0
        workers_hat = 0

        bounding_boxes = result.boxes

        for box in bounding_boxes:
            _, _, _, _, _, class_id = box.data.tolist()[0]
            workers_hat += 1 if class_id == 1 else 0
            drones_hat += 1 if class_id == 0 else 0

        for box in label:
            if len(box) == 0:
                continue

            workers += 1 if box[0] == 1 else 0
            drones += 1 if box[0] == 0 else 0

        drones_error += (drones - drones_hat) ** 2
        workers_error += (workers - workers_hat) ** 2

    drones_rmse = (drones_error / len(labels)) ** 0.5
    workers_rmse = (workers_error / len(labels)) ** 0.5

    return drones_rmse, workers_rmse


def parse_images_and_labels(data_path: str) -> Tuple[List[np.ndarray], List[str], List[List[List[float]]]] :
    """
    Parses the images and labels from the data directory.
    Args:
        data_path: data_path to the data directory, assumes that the data directory has the following structure:
        data
        ├── images
        │   ├── 000000.png
        ├── labels
        │   ├── 000000.txt

    Returns:
        Tuple[List[np.ndarray], List[str], List[List[List[float]]]]: a tuple containing the images, image filenames
        and labels.
    """
    image_filenames = os.listdir(os.path.join(data_path, "images"))
    images = [cv.imread(os.path.join(data_path, "images", image_path)) for image_path in image_filenames]
    label_filenames = [os.path.join(data_path, "labels", image_path.replace(".png", ".txt"))
                        for image_path in image_filenames]
    label_list = []

    for label_filename in label_filenames:
        with open(label_filename, "r") as f:
            lines = f.readlines()
            image_label = []
            for line in lines:
                bee_label = line.split(" ")
                bee_label = [float(x) for x in bee_label]
                image_label.append(bee_label)

            label_list.append(image_label)
    return images, image_filenames, label_list

if __name__ == "__main__":
    data_path = "/home/bee/bee-detection/data_appmais_lab/AppMAIS1s_labeled_data/train/"

    model_11s = ultralytics.YOLO("/home/bee/bee-detection/trained_on_11r_2022.pt")
    # model_1s = ultralytics.YOLO("/home/bee/bee-detection/trained_on_11r_2022.pt")

    images, images_filenames, labels_list = parse_images_and_labels(data_path)

    output_directory = os.path.abspath('/home/bee/bee-detection/model_and_label_outputs/')
    log_likelihoods_11s = beta_binom_on_data(model_11s, images, labels, images_filenames, copy_images_and_labels=True,
                                             output_dir=None)
    print(f'The mean log likelihood is {np.mean(log_likelihoods_11s)} from the model trained on the 11s data.')
    sorted_likelihoods = sorted(zip(log_likelihoods_11s, images_filenames), key=lambda x: x[0], reverse=True)
    print(sorted_likelihoods)
    # log_likelihoods_1s = beta_binom_on_data(model_1s, images, labels_list)
    # print(f'On the 11s val set, the mean log likelihood is {np.mean(log_likelihoods_11s)} from the model trained on the '
    #       f'11s data. The mean log likelihood is {np.mean(log_likelihoods_1s)} from the model trained on the 1s data.')

    drones_rmse_11s, workers_rmse_11s = rmse_on_data(model_11s, images, labels, images_filenames, copy_images_and_labels=True, output_dir = None)
    print(f"drone rmse: {drones_rmse_11s}, \nworker rmse: {workers_rmse_11s}")
