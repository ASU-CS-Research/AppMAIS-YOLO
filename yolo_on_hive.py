import ultralytics
import os
import shutil
import load_yolo_model as lym
from sklearn.model_selection import train_test_split

# metrics_map = ""
# label_list = os.listdir("./videos")
hives = ['2RB', '2L', '5R', '11R', '16LB', '12L', '4LB', '6RC', '10LB', '10RB', '16R', '8LB', '7R', '1L', '1R', '9R', '3RB', '3LB', '5LB', '12R', '8R', '13R', '13L', '14L', '14R', '7L', '9LB', '6L', "11L"]

# for name in label_list:
#     if name == "classes.txt" or len(name) < 5:
#         continue
#     subname = name[7:10]
#     # print(subname)
#     if '@' in subname:
#         subname = subname[:2]
#     hives.append(subname)
    # shutil.copyfile(f"./videos/{name}",f"./videos/{subname}/{name}")

#getting a list of hives
# for name in label_list:
#     if name == "classes.txt" or len(name) < 5:
#         continue
#     subname = name[7:11]
#     print(subname)
#     if '@' in subname:
#         subname = subname[:2]
#     hives.append(subname)

#print(hives)


#sorting, will mess up for 2 digit hives
def sorting_by_hive( src, dest):
    # dest_path = "/home/bee/bee-detection/data_appmais_lab/data_by_hive/"
    # src_path = "/home/bee/bee-detection/data_appmais_lab/"

    src_path = src
    dest_path = dest
    labels = os.listdir(f"{src_path}labels")


    for label in labels:
        for hive in hives:
            if hive in label and ("1" + hive not in label):
                file_name = label[:-4]

                # put the labeled file in the respective hive folder
                shutil.copyfile(f"{src_path}labels/{label}", f"{dest_path}{hive}/labels/{label}")

                # put the image file in the respective hive folder
                shutil.copyfile(f"{src_path}images/{file_name}.png", f"{dest_path}{hive}/images/{file_name}.png")
    #             # print(file_name)

#correcting sorting mistakes
# for hive in ["1L", "1R", "2L"]:
#     src_path = f"/home/bee/bee-detection/data_appmais_lab/data_by_hive/{hive}"
#     dest_path = "/home/bee/bee-detection/data_appmais_lab/data_by_hive/"
#     pics = os.listdir(f"{src_path}/images/")
#     for pic in pics:
#         testing = "1" + hive
#         if testing in pic:
#             file_name = pic[:-4]
#             print("executing for " + pic)
#                         # put the image file in the respective hive folder
#             shutil.move(f"{src_path}/images/{pic}", f"{dest_path}1{hive}/images/{pic}")
#
#                         # put the label file in the respective hive folder
#             shutil.move(f"{src_path}/labels_list/{file_name}.txt", f"{dest_path}1{hive}/labels_list/{file_name}.txt")
#                         # print(file_name)

# making the folders
def format_hives_directory(dest):
    for hive in hives:
        # data = f"/home/bee/bee-detection/data_appmais_lab/data_by_hive"
        data = f"{dest}/data_by_hive"
        os.mkdir(f"{data}/{hive}")
        os.mkdir(f"{data}/{hive}/images/")
        os.mkdir(f"{data}/{hive}/labels/")

#writing the yaml files, DOUBLE CHECK THE PATHS
def write_yamls(dest_path, val_path):
    for hive in hives:
        data_path = f"{dest_path}{hive}/"
        yaml = f"train: /home/bee/bee-detection/data/train/images\nval: {val_path}\ntest: /home/bee/bee-detection/data/test/images\n\nnc: 2\nnames: ['Drone', 'Worker']"
        output_filepath = f"{dest_path}{hive}/{hive}data.yaml"
        yaml_file = open(output_filepath, "w")
        # print(output_filepath)
        yaml_file.write(yaml)
        yaml_file.close()

# print(label_list)


#running yolo on a video/hive data and gettimg the metrics
def metrics_on_hives(path, val_hives = hives):
    pretrained_weights = os.path.abspath("/home/olofintuyita/AppMAIS-YOLO/runs/detect/train7/weights/best.pt")
    model = ultralytics.YOLO(model=pretrained_weights)
    # model.train(data='data.yaml', epochs=2, imgsz=(480, 640), verbose=True, batch=8, lr0=0.001)

    hive_metrics = {}

    for hive in val_hives:
        if not path:
            path = f"/home/olofintuyita/AppMAIS-YOLO/data_by_hive/{hive}/{hive}data.yaml"
            data_yaml = os.path.abspath(path)
        else:
            data_yaml = os.path.abspath(f"{path}/{hive}/{hive}data.yaml")
        metrics = model.val(data_yaml)
        hive_metrics[hive] = metrics
        model.export()

    for key in hive_metrics.keys():
        print(f"\n\n{key}\n\n{hive_metrics[key]}\n\n")



#lym.run_predictions_on_video(model=model, video_filepath="/home/olofintuyita/AppMAIS-YOLO/videos/AppMAIS7L@2023-06-26@11-55-00.h264", destination_video_path=f"output{hive}.mp4", show=False)

if __name__ == '__main__':
    _ = ""
    
