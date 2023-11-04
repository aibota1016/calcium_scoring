import utils 
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json





def get_2Dpred_labels(pred_labels_path):
    pred_labels = sorted(os.listdir(pred_labels_path))
    data = []
    for label in pred_labels:
        lines = utils.read_label_txt(os.path.join(pred_labels_path, label))
        patient_name = label.split('_')[0]
        slice_num =  label.split('.')[0].split('_')[-1]
        x,y,w,h = lines[-1][1:]
        data.append([patient_name, int(slice_num), float(x), float(y), float(w), float(h)])
    labels_df = pd.DataFrame(data, columns=['patient', 'slice', 'x', 'y', 'w', 'h'])
    return labels_df

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

    return labels_df


def map2Dto3D(pred_labels_path, ct_images_path):
    labels_df = average_pred_labels(get_2Dpred_labels(pred_labels_path))
    patients = labels_df['patient'].unique()
    losses = {}
    for patient in patients:
        x_pred = list(labels_df[labels_df['patient'] == patients[1]]['x'])[0]
        y_pred = list(labels_df[labels_df['patient'] == patient]['y'])[0]
        w_pred = list(labels_df[labels_df['patient'] == patient]['w'])[0]
        h_pred = list(labels_df[labels_df['patient'] == patient]['h'])[0]
        pred_labels = [x_pred, y_pred, w_pred, h_pred]
        idxs = list(labels_df[labels_df['patient'] == patient]['slice'])
        pred_center, pred_size = map2Dto3D_single_patient(ct_images_path, patient, pred_labels, idxs)
        true_center, true_size = utils.read_json(os.path.join(ct_images_path, patient, 'bifurcation.json'))
        write_to_json(os.path.join(ct_images_path, patient, 'bifurcation.json'), pred_center, pred_size, out_path=os.path.join(ct_images_path, patient, 'pred.json'))
        losses[patient] = calculate_loss(pred_center, pred_size, true_center, true_size)
    return losses
   

def write_to_json(json_path, pred_center, pred_size, out_path):
    with open(json_path) as f:
        label_json = json.load(f)
        label_json['markups'][0]['center'] = list(pred_center)
        label_json['markups'][0]['size'] = pred_size
        json_dumps= json.dumps(label_json, indent=4, separators=(",", ":"))
    with open(out_path, "w") as outfile:
        outfile.write(json_dumps)

def map2Dto3D_single_patient(ct_images_path, patient, pred_labels, idxs):
    # reverses the utils.get_yololabel_from_3Dmarkup function
    ct_im, spacing, origin, direction = utils.read_nifti_image(os.path.join(ct_images_path, patient, 'og_ct.nii'), only_img=False)
    im_h, im_w = ct_im.shape[1:]
    # [0.440348, 0.550781, 0.030599, 0.031901]
    x,y,w,h = utils.denormalize_bbox(pred_labels, [im_h, im_w])
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
    center = center_x, center_y, center_z
    size = w_true, h_true, l
    return center, size


def calculate_loss(pred_center, pred_size, true_center, true_size):
    array1 = np.array(pred_center + pred_size) 
    array2 = np.array(true_center + true_size)
    mse = ((array1 - array2) ** 2).mean()
    return mse




""" 

def group_slices_by_patient(pred_labels_path):
    temp = pred_labels_path.parent/'labels_grouped'
    temp.mkdir(parents=True, exist_ok=True)
    Path(temp/patient).mkdir(parents=True, exist_ok=True)
    pred_labels = sorted(os.listdir(pred_labels_path))
    patient = pred_labels[0].split('_')[0]
    for i in range(1, len(pred_labels)):
        current = str(pred_labels[i]).split('_')[0]
        if current != patient:
            patient = current
            patient_files.append(patient)
        Path(temp/patient/'images').mkdir(parents=True, exist_ok=True)
        Path(temp/patient/'labels').mkdir(parents=True, exist_ok=True)
        if not os.path.exists(temp/patient/'images'/images[0]):
            shutil.copy(os.path.join(dataset_path/'images', images[0]), temp/patient/'images'/images[0])
            shutil.copy(os.path.join(dataset_path/'labels', labels[0]), temp/patient/'labels'/labels[0])
        shutil.copy(os.path.join(dataset_path/'images', images[i]), temp/patient/'images'/images[i])
        shutil.copy(os.path.join(dataset_path/'labels', labels[i]), temp/patient/'labels'/labels[i])
"""



if __name__ == '__main__':
    path = '/Users/aibotasanatbek/Desktop/FYP2/experiments/patient_split500/val2/labels'
    data_path = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/raw/annotated_data_bii'
    losses = map2Dto3D(pred_labels_path=path, ct_images_path=data_path)

    average = sum(losses.values()) / len(losses)

    print("Average:", average)


    """ 
    Output:
    {'PD001': 16.90474042337743, 'PD033': 2.9898951113646124, 'PD039': 29.688938835672136, 'PD044': 31.83831095902971, 'PD065': 40.038723528511625, 'PD076': 24.856048075322075, 'PD092': 0.8628076640062, 'PD106': 1.533145995943449, 'PD134': 30.494741925251674, 'PD143': 97.7229928427797, 'PD149': 34.02722092531605, 'PD168': 36.68821815199133, 'PD171': 11.264034629738342, 'PD195': 15.995465990251235, 'PD205': 11.92443010931162, 'PD237': 31.92379867194101, 'PD240': 17.511656040392392, 'PD272': 14.290886796845719, 'PD296': 9.990531681232651, 'PD308': 2.6688879197700945}
    Average: 23.160773813902455
    """
