import gradio as gr
import os
import torch
from timeit import default_timer as timer
from typing import Tuple, Dict

from model import create_effnetb2_model

with open("class_names.txt", "r") as f:
    class_names = [food.strip() for food in f.readlines()]

effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=101)

effnetb2.load_state_dict(
    torch.load(
        f="09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth",
        map_location=torch.device("cpu"),
    )
)


def predict(img) -> Tuple[Dict, float]:
    start = timer()

    img = effnetb2_transforms(img).unsqueeze(0)

    effnetb2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim=1)

    pred_labels_and_probs = {
        class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))
    }

    end = timer()
    pred_time = round(end - start, 4)

    return pred_labels_and_probs, pred_time


example_list = [["examples/" + example] for example in os.listdir("examples")]


app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predictions"),
        gr.Number(label="Prediction time (s)"),
    ],
    examples=example_list,
    title="FoodVision Big",
    description="An EFficientNetB2 feature extractor for 101 food classes",
    article="Created at [09_model_deployment]",
)

app.launch()
