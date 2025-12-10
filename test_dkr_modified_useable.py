import torch
import dsacstar
import cv2

print(f"CUDA test: {torch.cuda.is_available()}")
print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"CUDA device properties: {torch.cuda.get_device_properties(0)}")

print(f"DSAC* test: {dsacstar}")

print(f"OpenCV test: {cv2}")