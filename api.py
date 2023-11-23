from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, File, UploadFile
import torch
from PIL import Image

from model import create_effnetb2_model, pred

effnetb2, effnetb2_transforms, class_names = {}, {}, []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global effnetb2, effnetb2_transforms, class_names
    with open("class_names.txt", "r") as f:
        class_names = [food.strip() for food in f.readlines()]

    effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=101)
    effnetb2.load_state_dict(
        torch.load(
            f="09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth",
            map_location=torch.device("cpu"),
        )
    )
    effnetb2.eval()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    img = Image.open(file.file)
    pred_labels, pred_time = pred(img, effnetb2, effnetb2_transforms, class_names)
    return {
        "pred_labels": dict(
            sorted(pred_labels.items(), key=lambda x: x[1], reverse=True)[:5]
        ),
        "pred_time": pred_time,
    }
