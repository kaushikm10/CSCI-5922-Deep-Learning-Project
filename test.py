from tensorflow import keras
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os

model = tf.keras.models.load_model('models/model_1000_epochs.h5')
psnrs = 0
ssims = 0
files = os.listdir("dataset/val/low_res")

for fname in files:

    img = cv2.resize(cv2.imread(f"dataset/val/low_res/{fname}"), (256, 256)) / 255.
    original = cv2.resize(cv2.imread(f"dataset/val/high_res/{fname}"), (256, 256))

    high_res_img = model.predict(np.expand_dims(img, axis=0))[0]

    high_res_img = np.uint8(np.clip(high_res_img*255, 0, 255))

    psnrs += peak_signal_noise_ratio(original, high_res_img)

    # Calculate SSIM
    ssim_value_r = structural_similarity(original[:, :, 0], high_res_img[:, :, 0])
    ssim_value_g = structural_similarity(original[:, :, 1], high_res_img[:, :, 1])
    ssim_value_b = structural_similarity(original[:, :, 2], high_res_img[:, :, 2])

    ssims += (ssim_value_r + ssim_value_g + ssim_value_b)/3

print(f"PSNR: {psnrs/len(files):.2f} dB")
print(f"SSIM: {ssims/len(files):.4f}")