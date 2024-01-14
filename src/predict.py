from ultralytics import YOLO
import utils
from pathlib import Path
import json
import os
from postprocessing import map2Dto3D_single_patient, write_to_json, get_labels_df_txt, detect_LM_calcium, calculate_distance
import shutil
import subprocess
import numpy as np




def preprocess_detection_inference(ct_path, save_path):
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


def predict_bifurcation(trained_model, ct_path, save_path='/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/processed/temp'):
    preprocess_detection_inference(ct_path, save_path)
    model = YOLO(trained_model)
    model(save_path, imgsz=512, save_txt=True, save_conf=True, save=True, project='predictions', iou=0.4)
    shutil.rmtree(save_path)



def process_output(image_path, 
                   template_json='/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/raw/annotated_data_bii/PD003/bifurcation.json', 
                   output_folder='/Users/aibotasanatbek/Documents/projects/calcium_scoring/src/predictions/predict',
                   true_center=None):
    pred_labels_df, images = get_labels_df_txt(output_folder)
    if pred_labels_df is not None: 
        pred_labels = [pred_labels_df['x'][0], pred_labels_df['y'][0], pred_labels_df['w'][0], pred_labels_df['h'][0]]
        idxs = list(pred_labels_df['slice'])
        pred_center, pred_size = map2Dto3D_single_patient(image_path, pred_labels, idxs)
        write_to_json(template_json, pred_center, pred_size, out_path=os.path.join(output_folder, 'pred.json'))
        if true_center is not None:
            dist = calculate_distance(pred_center, true_center)
            return dist
        return images
    else:
        print("No bifurcation predictions found for the input data")
        return None


def get_segmentation(file_name, inference_base, patient_folder=None):
    data = {"testing": [], "training": []}
    if patient_folder is not None:
        data["testing"].append({"image": patient_folder+'/'+file_name})
    else:
        data["testing"].append({"image": file_name})
        #shutil.rmtree(inference_base)
    with open("inference_datalist.json", 'w') as json_file:
        json.dump(data, json_file, indent=4)


    inference_path = os.path.join(inference_base, file_name.split('.')[0:-1][0] + '/' + file_name.split('.')[0:-1][0] + '_seg.nii.gz')
    print("inference_base: ", inference_base)
    print("inference_path: ", inference_path)
    inference_cmd = r"""python -m aorta_segmentation.scripts.infer run --config_file "['aorta_segmentation/configs/hyper_parameters.yaml','aorta_segmentation/configs/network.yaml', 'aorta_segmentation/configs/transforms_train.yaml','aorta_segmentation/configs/transforms_validate.yaml', 'aorta_segmentation/configs/transforms_infer.yaml']" """
     
    if not os.path.exists(inference_path):
        try:
            print("inference_path doesnt exist, performing inference ... \n")
            # Run the command and capture the output
            result = subprocess.check_output(inference_cmd, shell=True, stderr=subprocess.STDOUT, text=True)
            print("Command output:", result)
        except subprocess.CalledProcessError as e:
            print("Error running command:", e)
            print("Command output (if any):", e.output)        
    
    return inference_path






def test_method(model_path, data_path, project_folder):
    output_folder= os.path.join(project_folder, 'src/predictions/predict')
    dist, calcification = {}, {}
    pred_path = os.path.join(output_folder, 'pred.json')
    for patient_folder in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, patient_folder)):
            image_path = os.path.join(data_path, patient_folder, 'og_ct.nii')
            #bifurcation detection
            true_center, true_size = utils.read_json(os.path.join(data_path, patient_folder, 'bifurcation.json'))
            predict_bifurcation(trained_model=model_path, ct_path=image_path)
            patient_dist = process_output(image_path, output_folder=output_folder, true_center=true_center)
            if patient_dist is not None:
                dist[patient_folder] = patient_dist
                bifurcation = utils.get_3Dcoor_from_markup(pred_path, image_path)
                #aorta segmentation
                data_list_json = os.path.join(output_folder, 'inference_datalist.json')
                if os.path.exists(data_list_json):
                    shutil.rmtree(data_list_json)
                inference_base = os.path.join(project_folder, 'src/aorta_segmentation/prediction_testing', patient_folder)
                aorta_segmentation_path =  get_segmentation("og_ct.nii", inference_base, patient_folder)
                #LM calcium detection
                connected_points = detect_LM_calcium(image_path, aorta_segmentation_path, bifurcation)
                if len(connected_points) > 0:
                    calcification[patient_folder] = len(connected_points)
            else: 
                dist[patient_folder] = 'NaN'
            #shutil.rmtree(output_folder)
            #
    return dist, calcification



    



if __name__ == '__main__':
    #im_path = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/raw/annotated_data_bii/PD002/og_ct.nii'
    #predict(trained_model=model_path, ct_path=im_path)
    #image_path = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/src/predictions/og_ct.nii'
    #images = process_output(image_path)


    project_folder = '/Users/aibotasanatbek/Documents/projects/calcium_scoring'
    model_path ='/Users/aibotasanatbek/Documents/projects/calcium_scoring/experiments/tuned/train/weights/best.pt'
    #data_path = os.path.join(project_folder, 'data/raw/annotated_data_bii')
    data_path = '/Users/aibotasanatbek/Desktop/data'
    
     
    # testing the method
    dist, calcification = test_method(model_path, data_path, project_folder=project_folder)
    print(dist)
    print("len(dist): ", len(dist))
    average = sum(dist.values()) / len(dist)
    print("Average distance: ", average)

    print("Patients with the LM calcification: ")
    print(calcification)
    

    #testing the aorta segmentation inference
    #base = os.path.join(project_folder, 'src/aorta_segmentation/prediction_testing', 'PD065')
    #inference_path = get_segmentation(file_name='og_ct.nii', inference_base='', patient_folder='PD065')
    #print(inference_path)
    




