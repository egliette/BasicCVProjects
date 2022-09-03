import os
import shutil
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
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

def one_hot_transform(label, num_classes):
    label_tensor = torch.Tensor([label]).long()
    return torch.nn.functional.one_hot(label_tensor, num_classes=num_classes)

def plot_predictions_vs_labels(images, labels, outputs, predictions, labels_map):
    """"
    Return figure which shows 5 images with corresponding labels and predictions
    @param images (Tensor): batch of input images
    @param labels (Tensor): batch of labels
    @param outputs (Tensor): outputs of model(images)
    @param predictions (Tensor): argmax of outputs
    @return fig (figure): figure which shows 5 images with corresponding labels and predictions
    """
    probs = F.softmax(outputs)
    fig = plt.figure(figsize=(10, 10))
    for i in range(5):
        ith_label = labels_map[labels[i].cpu().item()]
        ith_pred = labels_map[predictions[i].item()]
        ith_prob = probs[i][predictions[i]].item()
        ax = fig.add_subplot(1, 5, i+1, xticks=[], yticks=[])
        plt.tight_layout()
        plt.imshow(images[i].cpu().squeeze(), cmap="gray")
        ax.set_title(
            "Label {0}\nPred {1}: {2:.1f}".format(
                ith_label,
                ith_pred,
                ith_prob
            ),
            color="green" if predictions[i]==labels[i] else "red"
        )
    return fig
