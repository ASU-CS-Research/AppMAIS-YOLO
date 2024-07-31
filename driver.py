import ultralytics
import os
import betabinomial as bb

data_path = "/home/bee/bee-detection/data_appmais_lab/AppMAIS1s_labeled_data/test"

model_11s = ultralytics.YOLO("/home/bee/bee-detection/trained_on_11r_2022.pt")
# model_1s = ultralytics.YOLO("/home/bee/bee-detection/trained_on_11r_2022.pt")

images, images_filenames, labels_list = bb.parse_images_and_labels(data_path)

output_directory = os.path.abspath('/home/bee/bee-detection/model_and_label_outputs/')

drones_mae_11s, drones_n_11s, workers_mae_11s, workers_n_11s = bb.mae_on_data(
    model_11s, images, labels_list, images_filenames, copy_images_and_labels=True, output_dir=None
)

drones_ce_11s, drones_cen_11s, workers_ce_11s, workers_cen_11s = bb.count_error_on_data(
    model_11s, images, labels_list, images_filenames, copy_images_and_labels=True, output_dir=None
)

data = [drones_mae_11s, drones_n_11s, workers_mae_11s, workers_n_11s]
data2 = [drones_ce_11s, drones_cen_11s, workers_ce_11s, workers_cen_11s]

box_data = []
for i in range(len(drones_mae_11s)):
    box_data.append(drones_mae_11s[i])
    box_data.append(drones_n_11s[i])
    # put the elements of drones_ce_11s and workers_ce_11s into a list
    box_data2 = []
for i in range(len(drones_ce_11s)):
    box_data2.append(drones_ce_11s[i])
    box_data2.append(workers_ce_11s[i])





