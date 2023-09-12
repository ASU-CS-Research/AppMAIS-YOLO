from typing import List

import numpy as np
import ultralytics
import os
from betabinomial import beta_binom_on_data, parse_images_and_labels, rmse_on_data, get_caste_count_labels_results
from r_squared_bees import plot, r_squared

import natsort
from sklearn.model_selection import KFold

def move_files_and_create_yaml(fold_ind, dataset_path, train_indices, val_indices, img_filenames,
                               labels_filenames, exist_ok=False):

    imgs_path = os.path.join(dataset_path, 'images')
    labels_path = os.path.join(dataset_path, 'labels')

    fold_path = os.path.join(dataset_path, 'folds', f'fold_{fold_ind}')
    os.makedirs(fold_path, exist_ok=exist_ok)
    train_path = os.path.join(fold_path, 'train')
    os.makedirs(train_path, exist_ok=exist_ok)
    os.makedirs(os.path.join(train_path, 'images'), exist_ok=exist_ok)
    os.makedirs(os.path.join(train_path, 'labels'), exist_ok=exist_ok)
    val_path = os.path.join(fold_path, 'val')
    os.makedirs(val_path, exist_ok=exist_ok)
    os.makedirs(os.path.join(val_path, 'images'), exist_ok=exist_ok)
    os.makedirs(os.path.join(val_path, 'labels'), exist_ok=exist_ok)
    data_yaml = os.path.join(fold_path, 'data.yaml')
    # move train images and labels to fold_path/train
    for i in train_indices:
        os.system(f'cp {os.path.join(imgs_path, img_filenames[i])} {train_path}/images')
        os.system(f'cp {os.path.join(labels_path, labels_filenames[i])} {train_path}/labels')
    # move val images and labels to fold_path/val
    for i in val_indices:
        os.system(f'cp {os.path.join(imgs_path, img_filenames[i])} {val_path}/images')
        os.system(f'cp {os.path.join(labels_path, labels_filenames[i])} {val_path}/labels')
    # write data.yaml
    with open(data_yaml, 'w') as f:
        f.write(f'train: {train_path}\n')
        f.write(f'val: {val_path}\n')
        f.write('nc: 2\n')
        f.write('names: [\'Drone\', \'Worker\']\n')
    print(data_yaml)

def split_data(data_path, random_state=42, shuffle=True):

    images_path = os.path.join(data_path, 'images')
    labels_path = os.path.join(data_path, 'labels')
    # Split the data at the data data_path into 5 folds
    # For each fold, train the model on the other 4 folds and evaluate on the remaining fold
    kfold = KFold(n_splits=5, shuffle=shuffle, random_state=random_state)
    print(len(os.listdir(images_path)), " total images.")
    image_filenames = os.listdir(images_path)
    image_filenames = natsort.natsorted(image_filenames)
    label_filenames = os.listdir(labels_path)
    label_filenames = natsort.natsorted(label_filenames)
    folds = kfold.split(image_filenames)
    for fold_index, (train_index, val_index) in enumerate(folds):
        print(f'Fold {fold_index}')
        print(f'Train index ({len(train_index)} images): {train_index}')
        print(f'Val index ({len(val_index)} images): {val_index}')
        move_files_and_create_yaml(fold_index, data_path, train_index, val_index, image_filenames, label_filenames,
                                   exist_ok=False)

def compare_metrics(model_paths: List[str], val_set_paths: List[str]) -> List[float]:
    """
    Compare the log likelihoods of the models on the respective validation sets.
    Args:
        model_paths List[str]: A list of paths to the model weights to compare.
        val_set_paths List[str]: A list of paths to the respective val set for each trained model

    Returns:
        List[float]: A list in the order of the model paths of the mean log likelihoods of the models on the
        respective validation sets.
    """
    mean_log_likelihoods = []
    worker_rmse_metrics = []
    drone_rmse_metrics = []
    output_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(model_paths[0]))),
                                    'log_likelihoods_per_fold')
    os.makedirs(output_directory, exist_ok=True)
    workers_true_predicted = []
    drones_true_predicted = []
    for i, (model_path, val_set_path) in enumerate(zip(model_paths, val_set_paths)):
        fold_dirname = os.path.basename(os.path.dirname(model_path))
        model = ultralytics.YOLO(model_path)
        images, image_filenames, labels = parse_images_and_labels(val_set_path)
        predictions = model.predict(images, conf=0.64)
        for results, label, image_filename in zip(predictions, labels, image_filenames):
            (workers_true, drones_true), (workers_predicted, drones_predicted) = \
                get_caste_count_labels_results(label, results)
            workers_true_predicted.append((workers_true, workers_predicted))
            drones_true_predicted.append((drones_true, drones_predicted))
        log_likelihoods = beta_binom_on_data(
            model, images, labels, image_filenames, copy_images_and_labels=False,
            output_dir=os.path.join(output_directory, fold_dirname)
        )
        worker_rmse, drone_rmse = rmse_on_data(model, images, labels, image_filenames)
        mean_log_likelihoods.append(np.mean(log_likelihoods))
        worker_rmse_metrics.append(worker_rmse)
        drone_rmse_metrics.append(drone_rmse)
        print(f'{fold_dirname} mean log likelihood: {mean_log_likelihoods[-1]}')
        print(f'worker rmse: {worker_rmse_metrics[-1]}')
        print(f'drone rmse: {drone_rmse_metrics[-1]}\n')
    workers_true, workers_predicted = zip(*workers_true_predicted)
    drones_true, drones_predicted = zip(*drones_true_predicted)
    plot(workers_true, workers_predicted, "True Worker Count", "Predicted Worker Count",
         "Worker Count Predicted against True", "model: 5-fold cv",
         save_dest="workers_pred_v_true_cv.png", show=True)
    plot(workers_true, workers_predicted, "True Worker Count", "Predicted Worker Count",
         "Worker Count Predicted against True",
         f"model: 5-fold cv, r_squared: {r_squared(workers_true, workers_predicted)}",
         save_dest="workers_pred_v_true_cv.png", plot_x_e_y=True, show=True)
    plot(drones_true, drones_predicted, "True Drone Count", "Predicted Drone Count",
         "Drone Count Predicted against True",
         f"model: 5-fold cv, r_squared: {r_squared(drones_true, drones_predicted)}",
         save_dest="drones_pred_v_true_cv.png", plot_x_e_y=True, show=True)
    return mean_log_likelihoods


if __name__ == '__main__':
    # data_path = os.data_path.abspath('/home/bee/bee-detection/data_appmais_lab/AppMAIS11s_labeled_data/cv_dataset')
    # split_data(data_path, random_state=42, shuffle=True)
    folds_path = '/home/bee/bee-detection/data_appmais_lab/AppMAIS11s_labeled_data/cv_dataset/folds/'
    fold_model_paths = [os.path.join(folds_path, fold, 'best.pt') for fold in os.listdir(folds_path)]
    fold_val_set_paths = [os.path.join(folds_path, fold, 'val') for fold in os.listdir(folds_path)]
    print(fold_model_paths)
    compare_metrics(fold_model_paths, fold_val_set_paths)

