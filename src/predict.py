from ultralytics import YOLO
import utils
import numpy as np
from pathlib import Path
import json
import os
import pandas as pd
from postprocessing import map2Dto3D_single_patient, write_to_json



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



#TODO: change the output folder to get the latest predict folder
#TODO: change the upload folder in the app.py to get the user uploaded file path
#TODO: change the app.py to optionally display results according to the checkboxes
#TODO: clear the temp folder after the prediction is done

def process_output(sample_json='/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/raw/annotated_data_bii/PD002/bifurcation.json', output_folder='/Users/aibotasanatbek/Documents/projects/calcium_scoring/src/predictions/predict'):
    labels_folder = Path(output_folder) / 'labels'
    labels = [x for x in labels_folder.iterdir() if x.suffix == '.txt']
    if len(labels) == 0:
        print('No predictions found')
    else:
        data, images = [], []
        for label in labels:
            label = str(label)
            images.append(output_folder + "/" + label.split("/")[-1].split('.')[0]+'.png')
            patient_name = label.split('/')[-1].split('_')[0]
            slice_num = label.split('_')[-1].split('.')[0]
            with open(label, 'r') as f:
                labels_data = f.readlines()
            label = labels_data[0].strip().split()
            class_id, x, y, w, h, conf = map(float, label)
            x,y,w,h = utils.denormalize_bbox([x,y,w,h], [512, 512])
            data.append([patient_name, int(slice_num), x, y, w, h, conf])
        pred_labels_df = pd.DataFrame(data, columns=['patient', 'slice', 'x', 'y', 'w', 'h', 'conf_score']).sort_values(by='patient', ignore_index=True)

        patients = pred_labels_df['patient'].unique()
        data = []
        for patient in patients:
            x_pred = list(pred_labels_df[pred_labels_df['patient'] == patient]['x'])[0]
            y_pred = list(pred_labels_df[pred_labels_df['patient'] == patient]['y'])[0]
            w_pred = list(pred_labels_df[pred_labels_df['patient'] == patient]['w'])[0]
            h_pred = list(pred_labels_df[pred_labels_df['patient'] == patient]['h'])[0]
            pred_labels = [x_pred, y_pred, w_pred, h_pred]
            idxs = list(pred_labels_df[pred_labels_df['patient'] == patient]['slice'])
            ct_images_path = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/raw/annotated_data_bii'
            pred_center, pred_size = map2Dto3D_single_patient(ct_images_path, 'PD002', pred_labels, idxs)
            write_to_json(sample_json, pred_center, pred_size, out_path=os.path.join(output_folder, 'pred.json'))
        return images

    



if __name__ == '__main__':
    model_path = '/Users/aibotasanatbek/Desktop/FYP2/experiments/final2/final_tuned/train/weights/best.pt'
    im_path = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/raw/annotated_data_bii/PD002/og_ct.nii'
    #predict(trained_model=model_path, ct_path=im_path)
    images = process_output()


