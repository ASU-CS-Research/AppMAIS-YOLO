import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from loguru import logger
import natsort


def copy_files(image_filepaths, label_filepaths, output_directory, input_directory):
    for image_filename, label_filename in tqdm(zip(image_filepaths, label_filepaths)):
        copy_image_string = f'cp "{os.path.join(input_directory, "images", image_filename)}" ' \
                            f'"{os.path.join(output_directory, "images", image_filename)}"'
        copy_label_string = f'cp "{os.path.join(input_directory, "labels", label_filename)}" ' \
                            f'"{os.path.join(os.path.join(output_directory, "labels"), label_filename)}"'
        # print(copy_image_string)
        # print(copy_label_string)
        os.system(copy_image_string)
        os.system(copy_label_string)


if __name__ == '__main__':
    # Set up the input and output locations
    input_location = os.path.abspath('/home/bee/bee-detection/data_appmais_lab/AppMAIS11s_labeled_data/')
    current_image_location = os.path.join(input_location, 'images')
    current_label_location = os.path.join(input_location, 'labels')
    output_location = os.path.join(input_location, 'split_dataset')
    random_seed = 42

    # Create a list of all the files in the input location
    image_filenames = os.listdir(current_image_location)
    label_filenames = os.listdir(current_label_location)
    image_filenames = natsort.natsorted(image_filenames)
    label_filenames = natsort.natsorted(label_filenames)

    # Split the dataset 60 train / 20 val / 20 test
    train_images, test_images, train_labels, test_labels = train_test_split(
        image_filenames, label_filenames, test_size=0.4, random_state=random_seed
    )
    for image_filename, label_filename in zip(train_images, train_labels):
        if image_filename.replace('.png', '') != label_filename.replace('.txt', ''):
            raise ValueError('Image and label filenames do not match.'
                             f'Image filename: {image_filename}\n'
                             f'Label filename: {label_filename}')

    test_images, val_images, test_labels, val_labels = train_test_split(
        test_images, test_labels, test_size=0.5, random_state=random_seed
    )

    # Create the output directories
    output_train_location = os.path.join(output_location, 'train')
    output_val_location = os.path.join(output_location, 'val')
    output_test_location = os.path.join(output_location, 'test')
    for output_dir in [output_train_location, output_val_location, output_test_location]:
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    # Copy the files to the output directories
    logger.info('Copying files to output directories...')
    logger.info('Copying train files.')
    copy_files(train_images, train_labels, output_train_location, input_location)
    logger.info('Copying val files.')
    copy_files(val_images, val_labels, output_val_location, input_location)
    logger.info('Copying test files.')
    copy_files(test_images, test_labels, output_test_location, input_location)