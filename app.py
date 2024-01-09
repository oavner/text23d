import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_gif
from PIL import Image
import webbrowser
import os

pipeline = DiffusionPipeline.from_pretrained("openai/shap-e")

guidance_scale = 15.0
while True:
    prompt = input("Lemme Gif that for you: ")
    images = pipeline(prompt,guidance_scale=guidance_scale,num_inference_steps=64,).images
    gif_path = export_to_gif(images[0], "3d.gif")
    gif_path = '3d.gif'
    current_directory = os.getcwd()
    webbrowser.open('file://' + current_directory + gif_path, new=2)