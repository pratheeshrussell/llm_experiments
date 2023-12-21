import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

device = torch.device('cpu')

model = RealESRGAN(device, scale=4)
model.load_weights('models/RealESRGAN_x4.pth', download=True)

path_to_image = 'image/2023-12-21-14-02-56.jpg'
image = Image.open(path_to_image).convert('RGB')

sr_image = model.predict(image)

sr_image.save('image/2023-12-21-14-02-56_enhanced.jpg')
