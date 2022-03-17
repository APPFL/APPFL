"""
https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
"""
import time

import json
from appfl.misc.data import *

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import roc_curve, auc

import matplotlib
import matplotlib.pyplot as plt


""" Isabelle's DenseNet (the outputs of the model are probabilities of 1 class ) """
import importlib.machinery

loader = importlib.machinery.SourceFileLoader("MainModel", "./IsabelleTorch.py")
MainModel = loader.load_module()
loss_fn = torch.nn.BCELoss()

""" Test Data """
start_time = time.time()
with open("../datasets/PreprocessedData/deepcovid32_all_data.json") as f:
    test_data_raw = json.load(f)

test_dataset = Dataset(
    torch.FloatTensor(test_data_raw["x"]),
    torch.FloatTensor(test_data_raw["y"]).reshape(-1, 1),
)

dataloader = DataLoader(
    test_dataset,
    num_workers=0,
    batch_size=len(test_dataset),
    shuffle=False,
)
print("Loading_Time=", time.time() - start_time)


""" get model """
file = "./save_models/Covid_Binary_Isabelle_FedAvg_LR_0.005_Round_40.pt"
model = torch.load(file)
model.eval()

with torch.no_grad():
    for img, target in dataloader:
        img = img.to("cpu")
        target = target.to("cpu")

        y_test = target.cpu().numpy()
        pred_y1 = model(img)
        loss = round(loss_fn(pred_y1, target).item(), 5)

        y_score = pred_y1.cpu().numpy()

        pred = pred_y1.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = round(correct / len(target) * 100, 2)

        print("y_test=", y_test, " size=", len(y_test))
        print("y_score=", y_score)
        print("loss=", loss)
        print("correct=", correct)
        print("accuracy=", accuracy)

        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        # Plot all ROC curves
        plt.figure()
        lw = 2
        colors = [
            "darkgreen",
            "navy",
            "deeppink",
            "aqua",
            "darkorange",
            "cornflowerblue",
        ]
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label="area = {0:0.2f}".format(roc_auc),
        )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curves of class 1")
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig("./plot.png", dpi=300, bbox_inches="tight")
        print("elapsed_time=", time.time() - start_time)
