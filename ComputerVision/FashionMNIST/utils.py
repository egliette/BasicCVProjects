import os
import shutil
import torchvision
import pandas as pd

def download_fmnist_img_data(is_training=True, save_dir="data/"):
    """
    Download FashionMNIST data from PyTorch then convert them into .jpg data
    @param is_training (bool): training data or test data
    @param save_dir (str): directory to save train/test folder and annotation file
    """
    dataset = torchvision.datasets.FashionMNIST(root="fmnist_data",
                                                train=is_training,
                                                transform=None,
                                                download=True)
    type_path = "train/" if is_training else "test/"
    subpath = save_dir + type_path
    if not os.path.exists(subpath):
        os.makedirs(subpath)

    data = {"file_name": list(), "label": list()}
    for idx, (img, label) in enumerate(dataset):
        file_name = f"{idx:06d}.jpg"
        save_path = subpath + file_name
        img.save(save_path)
        data["file_name"].append(file_name)
        data["label"].append(label)

    annotation_df = pd.DataFrame(data)
    annotation_file_name = "train_annotation.csv" if is_training else "test_annotation.csv"
    annotation_df.to_csv(save_dir+annotation_file_name, index=False)

    # remove PyTorch data folder
    shutil.rmtree("fmnist_data", ignore_errors=True)

