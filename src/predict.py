from ultralytics import YOLO
from .utils import fix_direction

#Using ultralytics Python SDK:

def preprocess_inference(ct_path):
    slice_images = []
    ct_im = fix_direction(ct_path)
    for idx in range(ct_im.shape[0]):
        slice_data = ct_im[idx, :, :]
        img = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
        slice_images.append(img)
    return slice_images


def predict(trained_model, ct_path):
    images = preprocess_inference(ct_path)
    results = trained_model([images])
    for result in results:
        boxes = result.boxes
        print(result.speed)
    return boxes


