import matplotlib.pyplot as plt
from scipy.stats import betabinom, binom, beta
import numpy as np
from typing import List, Optional
import ultralytics
import os
import cv2 as cv


def get_model_results(model, images, labels=None, image_filenames=None, copy_images_and_labels=False, output_dir=None):
    results = model.predict(images)
    # if copy_images_and_labels and output_dir is not None:
    #     images_for_model_output = [np.copy(image) for image in images]
    #     images_for_label_output = [np.copy(image) for image in images]
    #     # Apply bounding boxes to images
    #     for model_image, label_image, image_filename, result, label in (
    #             zip(images_for_model_output, images_for_label_output, image_filenames, results, labels)):
    #         # Plot the model output
    #         bounding_boxes = result.boxes
    #         for box in bounding_boxes:
    #             x1, y1, x2, y2, conf, class_id = box.data.tolist()[0]
    #             cv.rectangle(model_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    #             cv.putText(model_image, str(conf), (int(x1), int(y1)), cv.FONT_HERSHEY_SIMPLEX, 1,
    #                        (255, 0, 0), 2)
    #         # Plot the human made labels
    #         for label in labels:
    #             # Label is given as
    #         # save the images
    #         model_output_path = os.path.join(output_dir, 'model_outputs', image_filename.replace('.png', '') + "_output.png")
    #         cv.imwrite(model_output_path, model_image)
    return results


def beta_binom_on_data(model, images: List[np.ndarray], labels: np.ndarray, image_filenames: List[str],
                       copy_images_and_labels: Optional[bool] = False) -> List[float]:
    results = get_model_results(model, images, copy_images_and_labels)
    log_likelyhoods = []

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

        print("Image filename: ", filename)
        print("n_predicted: ", n)
        print("a_predicted: ", a)
        print("b_predicted: ", b)
        print("p_predicted: ", p)
        print("n_true: ", n_true)
        print("a_true: ", a_true)
        print("b_true: ", b_true)
        print("p_true: ", p_true)

        #mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')

        mean, var, skew, kurt = betabinom.stats(n_scaled, a_scaled, b_scaled, moments='mvsk')

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
        # binom_prob = binom.pmf(x, n_scaled, p)
        print("beta_binom_prob: ", beta_binom_prob)
        # print("binom_prob: ", binom_prob)
        # plt.show()

        print("log likelihood beta binom: ", np.log(beta_binom_prob))
        # print("log likelihood binom: ", np.log(binom_prob))

        log_likelyhoods.append(np.log(beta_binom_prob))
        print()

    return log_likelyhoods

if __name__ == "__main__":
    path = "/home/bee/bee-detection/data_appmais_lab/AppMAIS11s_labeled_data/split_dataset/val/"

    model_11s = ultralytics.YOLO("/home/bee/bee-detection/trained_on_11s.pt")
    # model_1s = ultralytics.YOLO("/home/bee/bee-detection/trained_on_1s.pt")

    images_filenames = os.listdir(f"{path}images/")
    images = [cv.imread(os.path.join(path,"images",image_path)) for image_path in images_filenames]
    labels_filenames = [os.path.join(path,"labels",image_path.replace(".png",".txt"))
                        for image_path in images_filenames]
    labels = []

    for label_filename in labels_filenames:
        with open(label_filename, "r") as f:
            lines = f.readlines()
            image_label = []
            for line in lines:
                bee_label = line.split(" ")
                bee_label = [float(x) for x in bee_label]
                image_label.append(bee_label)

            labels.append(image_label)


    log_likelihoods_11s = beta_binom_on_data(model_11s, images, labels, images_filenames)
    print(f'The mean log likelihood is {np.mean(log_likelihoods_11s)} from the model trained on the 11s data.')
    # log_likelihoods_1s = beta_binom_on_data(model_1s, images, labels)
    # print(f'On the 11s val set, the mean log likelihood is {np.mean(log_likelihoods_11s)} from the model trained on the '
    #       f'11s data. The mean log likelihood is {np.mean(log_likelihoods_1s)} from the model trained on the 1s data.')
