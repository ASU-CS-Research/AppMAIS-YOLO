import os
from sklearn.model_selection import train_test_split


def copy_files(image_filepaths, label_filepaths, output_directory):
    for image_filename, label_filename in zip(image_filepaths, label_filepaths):
        os.system(
            f'cp "{os.path.join(current_image_location, image_filename)}" '
            f'"{os.path.join(output_directory, "images", image_filename)}"'
        )
        os.system(
            f'{os.path.join(current_label_location, label_filename)} '
            f'{os.path.join(os.path.join(output_directory, "labels"), label_filename)}'
        )


if __name__ == '__main__':
    # Set up the input and output locations
    input_location = os.path.abspath('')
    current_image_location = os.path.join(input_location, 'images')
    current_label_location = os.path.join(input_location, 'labels')
    output_location = os.path.join(input_location, 'split_dataset')
    random_seed = 42

    # Create a list of all the files in the input location
    image_filenames = os.listdir(current_image_location)
    label_filenames = os.listdir(current_label_location)

    # Split the dataset 60 train / 20 val / 20 test
    train_images, test_images, train_labels, test_labels = train_test_split(
        image_filenames, label_filenames, test_size=0.4, random_state=random_seed
    )

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
    copy_files(train_images, train_labels, output_train_location)
    copy_files(val_images, val_labels, output_val_location)
    copy_files(test_images, test_labels, output_test_location)