import utils 
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
from visualizations import viz
import matplotlib.pyplot as plt




def get_labels_df_json(json_path):
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
            pred_labels_df = pred_labels_df[pred_labels_df['conf_score']>0.70]
            return pred_labels_df
            #print(pd.DataFrame(label_json).sort_values(by='image_id').drop('category_id', axis=1))
    else:
        print(f"The file '{json_path}' does not exist.")


def get_labels_df_txt(output_folder):
    labels_folder = Path(output_folder) / 'labels'
    labels = [x for x in labels_folder.iterdir() if x.suffix == '.txt']
    data, images = [], []
    for label in labels:
        label = str(label)
        images.append(output_folder + "/" + label.split("/")[-1].split('.')[0]+'.png')
        patient_name = 'pred'
        slice_num = label.split('_')[-1].split('.')[0]
        with open(label, 'r') as f:
            labels_data = f.readlines()
        label = labels_data[0].strip().split()
        class_id, x, y, w, h, conf = map(float, label)
        x,y,w,h = utils.denormalize_bbox([x,y,w,h], [512, 512])
        data.append([patient_name, int(slice_num), x, y, w, h, conf])
    pred_labels_df = pd.DataFrame(data, columns=['patient', 'slice', 'x', 'y', 'w', 'h', 'conf_score']).sort_values(by='patient', ignore_index=True)
    pred_labels_df = pred_labels_df[pred_labels_df['conf_score']>0.70]
    pred_labels_df = transform_labels(pred_labels_df)
    return pred_labels_df, images




def transform_labels(labels_df):
    #labels_df = avg_labels(labels_df)
    labels_df = weighted_avg(labels_df)
    
    temp_df = pd.DataFrame()
    temp_df['slice_min'] = labels_df.groupby('patient')['slice'].min()
    temp_df['slice_max'] = labels_df.groupby('patient')['slice'].max()
    # Expand the template dataframe with slices for each patient
    temp_df['slice'] = temp_df.apply(lambda row: list(range(row['slice_min'], row['slice_max'] + 1)), axis=1)
    temp_df = temp_df.explode('slice').drop(['slice_min', 'slice_max'], axis=1)
    # Merge the template dataframe with the original dataframe and fill missing values
    labels_df = pd.merge(temp_df, labels_df, on=['patient', 'slice'], how='left').fillna(method='ffill')
    #labels_df.to_csv('labels_df.csv')
    return labels_df

def avg_labels(labels_df):
    labels_df['y'] = labels_df.groupby('patient')['y'].transform('mean')
    labels_df['y'] = labels_df.groupby('patient')['y'].transform('mean')
    labels_df['w'] = labels_df.groupby('patient')['w'].transform('mean')
    labels_df['h'] = labels_df.groupby('patient')['h'].transform('mean')
    return labels_df

def weighted_avg(labels_df):
    weighted_avg_x = labels_df.groupby('patient').apply(lambda group: (group['x'] * group['conf_score']).sum() / group['conf_score'].sum())
    # Update the original DataFrame with the new weighted average x values
    labels_df['x'] = labels_df['patient'].map(weighted_avg_x)

    weighted_avg_y = labels_df.groupby('patient').apply(lambda group: (group['y'] * group['conf_score']).sum() / group['conf_score'].sum())
    labels_df['y'] = labels_df['patient'].map(weighted_avg_y)

    weighted_avg_w = labels_df.groupby('patient').apply(lambda group: (group['w'] * group['conf_score']).sum() / group['conf_score'].sum())
    labels_df['w'] = labels_df['patient'].map(weighted_avg_w)

    weighted_avg_h = labels_df.groupby('patient').apply(lambda group: (group['h'] * group['conf_score']).sum() / group['conf_score'].sum())
    labels_df['h'] = labels_df['patient'].map(weighted_avg_h)

    return labels_df


def map2Dto3D(pred_labels_path, ct_images_path):
    labels_df = transform_labels(get_labels_df_json(pred_labels_path))
    patients = labels_df['patient'].unique()
    losses = {}
    data = []
    for patient in patients:
        if patient in os.listdir(ct_images_path):
            x_pred = list(labels_df[labels_df['patient'] == patient]['x'])[0]
            y_pred = list(labels_df[labels_df['patient'] == patient]['y'])[0]
            w_pred = list(labels_df[labels_df['patient'] == patient]['w'])[0]
            h_pred = list(labels_df[labels_df['patient'] == patient]['h'])[0]
            pred_labels = [x_pred, y_pred, w_pred, h_pred]
            idxs = list(labels_df[labels_df['patient'] == patient]['slice'])
            image_path = os.path.join(ct_images_path, patient, 'og_ct.nii')
            pred_center, pred_size = map2Dto3D_single_patient(image_path, pred_labels, idxs)
            true_center, true_size = utils.read_json(os.path.join(ct_images_path, patient, 'bifurcation.json'))
            write_to_json(os.path.join(ct_images_path, patient, 'bifurcation.json'), pred_center, pred_size, out_path=os.path.join(ct_images_path, patient, 'pred.json'))
            losses[patient] = calculate_distance(pred_center, true_center)
            data.append([patient, pred_center, pred_size, true_center, true_size])
    df = pd.DataFrame(data, columns=['patient', 'pred_center', 'pred_size', 'true_center', 'true_size'])
    #df.to_csv(os.path.join(os.path.dirname(pred_labels_path), 'test_df.csv'))
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

def map2Dto3D_single_patient(ct_image_path, pred_labels, idxs):
    # reverses the utils.get_yololabel_from_3Dmarkup function
    ct_im, spacing, origin, direction = utils.read_nifti_image(ct_image_path, only_img=False)
    im_h, im_w = ct_im.shape[1:]
    x,y,w,h = pred_labels
    z = (max(idxs) + min(idxs)) / 2
    w_true = w * spacing[0]
    h_true = h * spacing[1]
    l = (max(idxs) - min(idxs) + 1) * spacing[2]
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











def detect_LM_calcium(ct_image_path, aorta_path, bifurcation_point, dilation=1, tolerance=10):
    """ bifurcation_point is a list containinng [z_center, y_center, x_center]"""
    # Metadata information
    im = utils.fix_direction(ct_image_path)
    aorta_mask = utils.fix_direction(aorta_path)
    
    # Generate all possible coordinates in 3D space
    x, y, z = np.meshgrid(np.arange(im.shape[2]), np.arange(im.shape[1]), np.arange(im.shape[0]), indexing='ij')
    points = np.column_stack((z.flatten(), y.flatten(), x.flatten()))
    # Create an array of aorta_points using the coordinates where aorta_mask coverss
    aorta_indices = np.where(aorta_mask == 1)
    aorta_points = np.column_stack(aorta_indices).astype(int)
    
    z_slice = int(bifurcation_point[0])
    slices = list(range(z_slice - dilation, z_slice + dilation + 1))
    filtered_aorta_points = [aorta_point[1:] for aorta_point in aorta_points if aorta_point[0] == z_slice]
    #print(len(aorta_points))

    nn2D = nn_dist(filtered_aorta_points, bifurcation_point[1:]) # y, x
    nearest_neighbor = np.insert(nn2D, 0, z_slice)
    print("bifurcation point: ", bifurcation_point)
    print("nearest point of the aorta to the bifurcation point: ", nearest_neighbor) #z, y, x
    #viz.plot_single_slice(im[33])
    results3D = []
    for slice in slices:
        filtered_aorta_points = [aorta_point[1:] for aorta_point in aorta_points if aorta_point[0] == slice]
        nn2D = nn_dist(filtered_aorta_points, bifurcation_point[1:]) # y, x
        nn3D = np.insert(nn2D, 0, slice) #z, x, y
        line_start, line_end = determine_line_orientation(np.insert(bifurcation_point[1:], 0, slice), nn3D)
        filtered_points = [[point[1], point[2]] for point in points if point[0] == slice]
        result2D = points_close(filtered_points, line_start[1:], line_end[1:], tolerance)
        results3D += [np.insert(point, 0, slice) for point in result2D]
   # print("Number of points close to the line: ", len(results3D))

    #plot_3d_scatter(results3D, bifurcation_point=np.insert(bifurcation_point[1:], 0, int(bifurcation_point[0])), nearest_neighbor_point_3D=nearest_neighbor)
    connected_points = calcium_thresholding(im, results3D, threshold=130)
    return connected_points
    


def points_close(points_list, line_start, line_end, tolerance):
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    line_direction = line_end - line_start
    # Normalize the direction vector
    line_direction /= np.linalg.norm(line_direction)
    result_points = []
    for point in points_list:
        # Check if the point is within the range of the line segment
        if np.all(np.logical_and(point > line_start, point < line_end)):
            # Vector from line_start to the point
            vector_to_point = point - line_start
            # Calculate the projection of vector_to_point onto the line
            projection = np.dot(vector_to_point, line_direction) * line_direction
            # Calculate the distance between the point and its projection
            distance = np.linalg.norm(vector_to_point - projection)
            # Check if the point is close to the line within the specified tolerance
            if distance <= tolerance:
                result_points.append(point)
    return result_points
    
def determine_line_orientation(point1, point2):
    for dim in range(3):
        if point1[dim] < point2[dim]:
            return point1, point2
        elif point1[dim] > point2[dim]:
            return point2, point1


def nn_dist(points, bifurcation_point):
    """ Calculates the point with the nearest distance in the set of points from the given bifurcation point"""
    from scipy.spatial.distance import cdist
    distances = cdist(points, [bifurcation_point])
    nearest_idx = np.argmin(distances)
    nearest_neighbor = points[nearest_idx]
    return nearest_neighbor




def plot_3d_scatter(points, bifurcation_point, nearest_neighbor_point_3D, aorta_points=None):
    """
    Visualize a 3D scatter plot from a list of 3D points.

    Parameters:
    - points (list): List of tuples representing 3D points (x, y, z).
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    z, y, x = zip(*points)
    bz, by, bx = bifurcation_point
    nz, ny, nx = nearest_neighbor_point_3D
    # Plot the line
    ax.plot([bx, nx], [by, ny], [bz, nz], label="LM Line", color="black", alpha=0.5)

    ax.scatter(x,y,z, marker='o', alpha=0.2, color='orange', label='points close to LM line')
    ax.scatter(bx, by, bz, marker='o', alpha=1, color='red', label='Bifurcation point')
    ax.scatter(nx, ny, nz, marker='o', alpha=1, color='green', label='Nearest neighbor point')
    if aorta_points is not None:
        az, ay, ax = zip(*aorta_points)
        ax.scatter(ax, ay, az, marker='o', alpha=0.2, color='steelblue', label='Aorta points')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    #ax.set_zlim(20, 50)
    plt.legend()
    plt.show()





def calcium_thresholding(im, points, threshold=130):
    filtered_indexes = []
    for index in points:
        z, y, x = index
        # Check if the value at the given index is above the threshold
        if im[z, y, x] > threshold:
            filtered_indexes.append((z, y, x))
    #print("Points with values above threshold:", filtered_indexes)
    #print("Number of points with values above threshold:", len(filtered_indexes))
    connected_points = set()
    # Check for 6-connectivity among the selected points
    for point in filtered_indexes:
        z, y, x = point
        # Check the 6-connectivity in the neighborhood
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    if (i != 0 or j != 0 or k != 0) and 0 <= z + i < im.shape[0] and 0 <= y + j < im.shape[1] and 0 <= x + k < im.shape[2]:
                        neighbor_index = (z + i, y + j, x + k)
                        if im[neighbor_index] > threshold:
                            connected_points.add(neighbor_index)
    #print("Points with values above threshold and 6-connectivity:", connected_points)
    #print("Number of points with values above threshold and 6-connectivity:", len(connected_points))
    return list(connected_points)



if __name__ == '__main__':

    data_path = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/raw/annotated_data_bii/PD071/og_ct.nii'
    aorta_path = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/raw/annotated_data_bii/PD071/aorta_mask.nii'
    markup_path = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/raw/annotated_data_bii/PD071/bifurcation.json'

    """ 
    losses = map2Dto3D(pred_labels_path=path, ct_images_path=data_path)
    print(losses)
    average = sum(losses.values()) / len(losses)
    print("Average:", average)
    """

    

    bifurcation = utils.get_3Dcoor_from_markup(markup_path, data_path)
    connected_points = detect_LM_calcium(data_path, aorta_path, bifurcation)
    print(len(connected_points))


    data_folder = '/Users/aibotasanatbek/Desktop/data'
    """ 
    patients_with_LM_calcium = []
    for patient in os.listdir(data_folder):
        if os.path.isdir(os.path.join(data_folder, patient)):
            patient_ct_file = os.path.join(data_folder, patient, 'og_ct.nii')
            patient_aorta_path = os.path.join(data_folder, patient, 'aorta_mask.nii')
            patient_bifurcation_path = os.path.join(data_folder, patient, 'bifurcation.json')
            bifurcation = utils.get_3Dcoor_from_markup(patient_bifurcation_path, patient_ct_file)
            connected_points = get_LM(patient_ct_file, patient_aorta_path, bifurcation)
            if len(connected_points) > 0:
                data = {}
                data['patient_name'] = patient 
                data['length'] = len(connected_points)
                patients_with_LM_calcium.append(data)
    print(patients_with_LM_calcium)
    print(len(patients_with_LM_calcium))
    """







