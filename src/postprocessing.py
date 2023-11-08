import utils 
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json




def get_2Dpred_labels(json_path):
    if os.path.exists(json_path):
        data = []
        with open(json_path) as f:
            pred_labels_json = json.load(f)
            for label in pred_labels_json:
                patient_name = label['image_id'].split('_')[0]
                x,y,w,h = label['bbox']
                slice_num = label['image_id'].split('_')[-1]
                conf_score = label['score']
                data.append([patient_name, int(slice_num), x, y, w, h, conf_score])
            pred_labels_df = pd.DataFrame(data, columns=['patient', 'slice', 'x', 'y', 'w', 'h', 'conf_score']).sort_values(by='patient', ignore_index=True)
            pred_labels_df = pred_labels_df[pred_labels_df['conf_score']>0.80]
            return pred_labels_df
            #print(pd.DataFrame(label_json).sort_values(by='image_id').drop('category_id', axis=1))
    else:
        print(f"The file '{json_path}' does not exist.")
"""

def get_2Dpred_labels(pred_labels_path):
    pred_labels = sorted(pred_labels_path.rglob("*.txt"))
    data = []
    for label in pred_labels: 
        patient_name = str(label).split('/')[-1].split('_')[0]
        lines = utils.read_label_txt(os.path.join(pred_labels_path, label))
        if lines != "":
            slice_num =  str(label).split('.')[0].split('_')[-1]
            x,y,w,h = lines[-1][1:]
        data.append([patient_name, int(slice_num), float(x), float(y), float(w), float(h)])
    labels_df = pd.DataFrame(data, columns=['patient', 'slice', 'x', 'y', 'w', 'h'])
    return labels_df
"""


def average_pred_labels(labels_df):
    labels_df['x'] = labels_df.groupby('patient')['x'].transform('mean')
    labels_df['y'] = labels_df.groupby('patient')['y'].transform('mean')
    labels_df['w'] = labels_df.groupby('patient')['w'].transform('mean')
    labels_df['h'] = labels_df.groupby('patient')['h'].transform('mean')
    
    temp_df = pd.DataFrame()
    temp_df['slice_min'] = labels_df.groupby('patient')['slice'].min()
    temp_df['slice_max'] = labels_df.groupby('patient')['slice'].max()
    
    # Expand the template dataframe with slices for each patient
    temp_df['slice'] = temp_df.apply(lambda row: list(range(row['slice_min'], row['slice_max'] + 1)), axis=1)
    temp_df = temp_df.explode('slice').drop(['slice_min', 'slice_max'], axis=1)
    # Merge the template dataframe with the original dataframe and fill missing values
    labels_df = pd.merge(temp_df, labels_df, on=['patient', 'slice'], how='left').fillna(method='ffill')
    labels_df.to_csv('labels_df.csv')
    return labels_df


def map2Dto3D(pred_labels_path, ct_images_path):
    labels_df = average_pred_labels(get_2Dpred_labels(pred_labels_path))
    patients = labels_df['patient'].unique()
    losses = {}
    data = []
    for patient in patients:
        if patient in os.listdir(ct_images_path):
            x_pred = list(labels_df[labels_df['patient'] == patients[1]]['x'])[0]
            y_pred = list(labels_df[labels_df['patient'] == patient]['y'])[0]
            w_pred = list(labels_df[labels_df['patient'] == patient]['w'])[0]
            h_pred = list(labels_df[labels_df['patient'] == patient]['h'])[0]
            pred_labels = [x_pred, y_pred, w_pred, h_pred]
            idxs = list(labels_df[labels_df['patient'] == patient]['slice'])
            pred_center, pred_size = map2Dto3D_single_patient(ct_images_path, patient, pred_labels, idxs)
            true_center, true_size = utils.read_json(os.path.join(ct_images_path, patient, 'bifurcation.json'))
            write_to_json(os.path.join(ct_images_path, patient, 'bifurcation.json'), pred_center, pred_size, out_path=os.path.join(ct_images_path, patient, 'pred.json'))
            losses[patient] = calculate_distance(pred_center, true_center)
            data.append([patient, pred_center, pred_size, true_center, true_size])
    df = pd.DataFrame(data, columns=['patient', 'pred_center', 'pred_size', 'true_center', 'true_size'])
    df.to_csv(os.path.join(os.path.dirname(pred_labels_path), 'test_df.csv'))
    return losses
   

def write_to_json(json_path, pred_center, pred_size, out_path):
    with open(json_path) as f:
        label_json = json.load(f)
        label_json['markups'][0]['center'] = pred_center
        label_json['markups'][0]['size'] = pred_size
        label_json['markups'][0]['controlPoints'][0]['position'] = pred_center
        json_dumps= json.dumps(label_json, indent=4, separators=(",", ":"))
    with open(out_path, "w") as outfile:
        outfile.write(json_dumps)

def map2Dto3D_single_patient(ct_images_path, patient, pred_labels, idxs):
    # reverses the utils.get_yololabel_from_3Dmarkup function
    ct_im, spacing, origin, direction = utils.read_nifti_image(os.path.join(ct_images_path, patient, 'og_ct.nii'), only_img=False)
    im_h, im_w = ct_im.shape[1:]
    x,y,w,h = pred_labels
    z = (max(idxs) + min(idxs)) / 2
    w_true = w * spacing[0]
    h_true = h * spacing[1]
    l = (max(idxs) - min(idxs)) * spacing[2]
    flip_axes = [i for i, val in enumerate(np.diag(direction)) if val < 0]
    if 1 in flip_axes:
        y = y - im_h
    if 0 in flip_axes:
        x = x - im_w/2
    center_x = x * spacing[0] + origin[0]
    center_y = y * spacing[1] + origin[1]
    center_z = z * spacing[2] + origin[2]
    center = [center_x, center_y, center_z]
    size = [w_true, h_true, l]
    return center, size


def calculate_distance(pred_center, true_center):
    distance = np.sqrt(np.sum((np.array(true_center) - np.array(pred_center))**2, axis = 0))
    return distance






if __name__ == '__main__':
    path = '/Users/aibotasanatbek/Desktop/val3/predictions.json'
    data_path = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/raw/annotated_data_bii'
     
    losses = map2Dto3D(pred_labels_path=path, ct_images_path=data_path)
    print(losses)
    average = sum(losses.values()) / len(losses)
    print("Average:", average)
    