import os

import cv2
from tqdm import tqdm


# Utility file to convert relative center xywh label format to absolute xywh without changing the labels


def find_all_labels(image_width, image_height, label_path):
    combined_new_labels = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            label = line.split()
            label[0] = int(label[0])
            label[1] = float(label[1]) * image_width
            label[2] = float(label[2]) * image_height
            label[3] = float(label[3]) * image_width
            label[4] = float(label[4]) * image_height
            # label = [str(x) for x in label]
            # print(label)
            new_labels = [
                label[0],
                int(label[1]) - int(label[3]) // 2,  # x
                int(label[2]) - int(label[4]) // 2,  # y
                int(label[1]) + int(label[3]) // 2,  # w
                int(label[2]) + int(label[4]) // 2   # h
            ]
            combined_new_labels.append(new_labels)

    return combined_new_labels


def write_to_file(label_path, combined_new_labels):
    print(f'Writing new labels to {label_path}')
    with open(label_path, 'w') as f:
        for label in combined_new_labels:
            label = [str(x) for x in label]
            # print(' '.join(label) + '\n')
            f.write(' '.join(label) + '\n')


if __name__ == '__main__':
    image_path = 'data/train/images/0.jpg'
    labels_path = 'data/test/labels'
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape
    label_documents = os.listdir(labels_path)
    print(f'running on {len(label_documents)} label documents...')
    for label_document in tqdm(label_documents):
        new_labels = find_all_labels(image_width, image_height, os.path.join(labels_path, label_document))
        ### ### ALREADY DONE, DO NOT RUN AGAIN ### ###
        # write_to_file(os.path.join(labels_path, label_document), new_labels)
