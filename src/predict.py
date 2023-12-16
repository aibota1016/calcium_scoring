from ultralytics import YOLO
import utils
import numpy as np
from pathlib import Path
import json
import os
import pandas as pd
from postprocessing import map2Dto3D_single_patient, write_to_json, get_labels_df_txt, detect_LM_calcium, calculate_distance
import shutil



def preprocess_inference(ct_path, save_path):
    slice_images = []
    ct_im = utils.fix_direction(ct_path)
    for idx in range(ct_im.shape[0]):
        slice_data = ct_im[idx, :, :]
        img = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
        # save as png
        image_path = Path(save_path) / f'{ct_path.split("/")[-2]}_{str(idx)}.png'
        image_path.parent.mkdir(parents=True, exist_ok=True)
        utils.save_array_as_png(img, image_path)
        slice_images.append(img)


def predict(trained_model, ct_path, save_path='/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/processed/temp'):
    preprocess_inference(ct_path, save_path)
    model = YOLO(trained_model)
    model(save_path, imgsz=512, save_txt=True, save_conf=True, save=True, project='predictions', iou=0.4)
    shutil.rmtree(save_path)

    
    


def process_output(image_path, 
                   template_json='/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/raw/annotated_data_bii/PD003/bifurcation.json', 
                   output_folder='/Users/aibotasanatbek/Documents/projects/calcium_scoring/src/predictions/predict',
                   true_center=None):
    pred_labels_df, images = get_labels_df_txt(output_folder)
    pred_labels = [pred_labels_df['x'][0], pred_labels_df['y'][0], pred_labels_df['w'][0], pred_labels_df['h'][0]]
    idxs = list(pred_labels_df['slice'])
    pred_center, pred_size = map2Dto3D_single_patient(image_path, pred_labels, idxs)
    write_to_json(template_json, pred_center, pred_size, out_path=os.path.join(output_folder, 'pred.json'))
    if true_center is not None:
        dist = calculate_distance(pred_center, true_center)
        return dist
    return images




#TODO: integrate aorta segmentation model
def test_method(model_path, data_path, output_folder):
    dist, calcification = {}, {}
    pred_path = os.path.join(output_folder, 'pred.json')
    for patient_folder in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, patient_folder)):
            image_path = os.path.join(data_path, patient_folder, 'og_ct.nii')
            aorta_filepath = os.path.join(data_path, patient_folder, 'aorta_mask.nii')
            true_center, true_size = utils.read_json(os.path.join(data_path, patient_folder, 'bifurcation.json'))
            predict(trained_model=model_path, ct_path=image_path)
            patient_dist = process_output(image_path, output_folder=output_folder, true_center=true_center)
            dist[patient_folder] = patient_dist
            bifurcation = utils.get_3Dcoor_from_markup(pred_path, image_path)
            connected_points = detect_LM_calcium(image_path, aorta_filepath, bifurcation)
            if len(connected_points) > 0:
                calcification[patient_folder] = len(connected_points)
            shutil.rmtree(output_folder)
    return dist, calcification



    



if __name__ == '__main__':
    model_path = '/Users/aibotasanatbek/Desktop/FYP2/experiments/final2/final_tuned/train/weights/best.pt'
    #im_path = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/raw/annotated_data_bii/PD002/og_ct.nii'
    #predict(trained_model=model_path, ct_path=im_path)
    #image_path = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/src/predictions/og_ct.nii'
    #images = process_output(image_path)


    model_path ='/Users/aibotasanatbek/Desktop/FYP2/experiments/final2/final_tuned/train/weights/best.pt'
    #data_path = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/raw/annotated_data_bii'
    data_path = '/Users/aibotasanatbek/Desktop/data'
    output_folder='/Users/aibotasanatbek/Documents/projects/calcium_scoring/src/predictions/predict'

    dist, calcification = test_method(model_path, data_path, output_folder=output_folder)
    print(dist)
    print("len(dist): ", len(dist))
    average = sum(dist.values()) / len(dist)
    print("Average distance: ", average)

    print("Patients with the LM calcification: ")
    print(calcification)
    print("Number of patients with LM calcification: ", len(calcification))





