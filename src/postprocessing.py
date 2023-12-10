import utils 
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
from visualizations import viz
import matplotlib.pyplot as plt




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
            pred_labels_df = pred_labels_df[pred_labels_df['conf_score']>0.85]
            return pred_labels_df
            #print(pd.DataFrame(label_json).sort_values(by='image_id').drop('category_id', axis=1))
    else:
        print(f"The file '{json_path}' does not exist.")




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
    labels_df.to_csv('labels_df.csv')
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
    labels_df = transform_labels(get_2Dpred_labels(pred_labels_path))
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







def get_LM(ct_image_path, aorta_path, bifurcation_point, dilation=3, tolerance=4.0):
    """ bifurcation_point is a list containinng [x_center, y_center, z_center]"""
    # Metadata information
    im = utils.read_nifti_image(ct_image_path)
    aorta_mask = utils.read_nifti_image(aorta_path)

    # Generate all possible coordinates in 3D space
    x, y, z = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[2]), np.arange(im.shape[0]), indexing='ij')
    points = np.column_stack((z.flatten(), x.flatten(), y.flatten()))
    # Create an array of aorta_points using the coordinates where aorta_mask covers
    aorta_indices = np.where(aorta_mask == 0)
    print("Number of points in the aorta: ", len(aorta_indices[0]))
    aorta_points = np.column_stack(aorta_indices).astype(int)
    filtered_aorta_points = [point[1:] for point in aorta_points if point[0] == int(bifurcation_point[0])]
    print(filtered_aorta_points)

    


    nn2D = nn_dist(filtered_aorta_points, bifurcation_point[1:]) # x, y
    nearest_neighbor = np.insert(nn2D, 0, int(bifurcation_point[0]))
    print("bifurcation point: ", bifurcation)
    print("nearest point of the aorta to the bifurcation point: ", nearest_neighbor) #z, x, y

    viz.plot_single_slice(im[33])
    viz.plot_single_slice(aorta_mask[33])

    #plot_3d_scatter(aorta_points, bifurcation_point, nn_3D)
    
    #inside_points = [point for point in points if is_point_inside_cuboid(point, bifurcation, nearest_neighbor_point_3D, dilation_factor)]
    #plot_3d_scatter(inside_points, bifurcation_point, nearest_neighbor_point_3D)

    #nn2 = nn_dist_3D(aorta_points, bifurcation_point)
    #print("nearest point v2: ", nn2) #z, x, y
    results3D = []
    slices = list(range(int(bifurcation_point[0]) - dilation, int(bifurcation_point[0]) + dilation + 1))
    print(slices)
    """
    for slice in slices:
        nn3D = np.insert(nn2D, 0, slice) #z, x, y
        line_start, line_end = determine_line_orientation(np.insert(bifurcation_point[1:], 0, slice), nn3D)
        filtered_points = [[point[1], point[2]] for point in points if point[0] == slice]
        result2D = points_close(filtered_points, line_start[1:]-[5,5], line_end[1:]+[5,5], tolerance)
        #print(result2D)
        results3D += [np.insert(point, 0, slice) for point in result2D]
    print("Number of points close to the line: ", len(results3D))
    print(results3D[:5])
    #plot_3d_scatter(results3D, np.insert(bifurcation_point[1:], 0, int(bifurcation_point[0])), nearest_neighbor)
    #plot_2d_scatter(result2D, bifurcation_point[1:], nearest_neighbor_point)
    connected_points = calcium_thresholding(im, results3D, threshold=130)
    """
    


def points_close(points_list, line_start, line_end, tolerance=0.5):
    # Convert line_start, line_end, and line_direction to NumPy arrays
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    line_direction = line_end - line_start

    # Normalize the direction vector
    line_direction /= np.linalg.norm(line_direction)

    result_points = []

    for point in points_list:
        # Check if the point is within the range of the line segment
        if np.all(np.logical_and(point >= line_start, point <= line_end)):
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
    # Compare coordinates for each dimension (x, y, z)
    for dim in range(3):  # 0 for x, 1 for y, 2 for z
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

    """ 

    from scipy.spatial import cKDTree
    kdtree = cKDTree(points)
    _, nearest_idx = kdtree.query(bifurcation_point)
    nearest_neighbor = points[nearest_idx]
    return nearest_neighbor
    """

def nn_dist_3D(points, bifurcation_point):
    from scipy.spatial import cKDTree
    kdtree = cKDTree(points)
    _, nearest_idx = kdtree.query(bifurcation_point)
    nearest_neighbor = points[nearest_idx]
    return nearest_neighbor



def find_line_between_points(point1, point2):

    # Calculate the direction vector of the line
    line_direction = point2 - point1

    return point1, point2, line_direction


def points_on_line_or_close(points_list, line_start, line_direction, tolerance=0.1):
    # Convert line_start and line_direction to NumPy arrays
    line_start = np.array(line_start)
    line_direction = np.array(line_direction)

    # Normalize the direction vector
    line_direction /= np.linalg.norm(line_direction)

    result_points = []

    for point in points_list:
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








def is_point_inside_cuboid(point, start_point, end_point, dilation_factor):
    z1, x1, y1 = start_point
    z2, x2, y2 = end_point
    zp, xp, yp = point
    # Check conditions for point inclusion with dilation factor
    condition_x = (x1 <= xp <= x2 or x2 <= xp <= x1) and (xp - x1) / (x2 - x1) <= dilation_factor
    condition_y = (y1 <= yp <= y2 or y2 <= yp <= y1) and (yp - y1) / (y2 - y1) <= dilation_factor
    condition_z = (z1 <= zp <= z2 or z2 <= zp <= z1) and (zp - z1) / (z2 - z1) <= dilation_factor
    return condition_x and condition_y and condition_z



def plot_3d_scatter(points, bifurcation_point, nearest_neighbor_point_3D):
    """
    Visualize a 3D scatter plot from a list of 3D points.

    Parameters:
    - points (list): List of tuples representing 3D points (x, y, z).
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    z, x, y = zip(*points)
    bz, bx, by = bifurcation_point
    nz, nx, ny = nearest_neighbor_point_3D
    # Plot the line
    ax.plot([bx, nx], [by, ny], [bz, nz], label="LM Line", color="black", alpha=0.5)

    ax.scatter(x,y,z, marker='o', alpha=0.5, color='steelblue')
    ax.scatter(bx, by, bz, marker='o', alpha=1, color='red', label='Bifurcation point')
    ax.scatter(nx, ny, nz, marker='o', alpha=1, color='green', label='Nearest neighbor point')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_zlim(20, 50)
    plt.legend()
    plt.show()




def plot_2d_scatter(points_list, bifurcation_point2D, nn2D):
    # Unpack the x and y coordinates from the list of points
    x_coordinates, y_coordinates = zip(*points_list)
    bx, by = bifurcation_point2D
    nx, ny = nn2D

    # Plot the line
    plt.plot([bx, nx], [by, ny], label="LM Line", color="black", alpha=0.5)

    # Plot the scatter plot
    plt.scatter(x_coordinates, y_coordinates, marker='o', alpha=0.5, color='steelblue')
    plt.scatter(bx, by, marker='o', alpha=1, color='red', label='Bifurcation point')
    plt.scatter(nx, ny, marker='o', alpha=1, color='green', label='Nearest neighbor point')

    # Set labels and title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("2D Scatter Plot")

    # Remove x and y ticks
    #plt.xticks([])
    #plt.yticks([])

    # Display the legend
    plt.legend()

    # Show the plot
    plt.show()


def calcium_thresholding(im, points, threshold=130):
    filtered_indexes = []
    for index in points:
        z, x, y = index
        #print(im[z, x, y])
        # Check if the value at the given index is above the threshold
        if im[z, x, y] > threshold:
            filtered_indexes.append((z, x, y))
    print("Points with values above threshold:", filtered_indexes)
    connected_points = set()
    # Check for 6-connectivity among the selected points
    for point in filtered_indexes:
        z, x, y = point
        # Check the 6-connectivity in the neighborhood
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    if (i != 0 or j != 0 or k != 0) and 0 <= x + i < im.shape[0] and 0 <= y + j < im.shape[1] and 0 <= z + k < im.shape[2]:
                        neighbor_index = (x + i, y + j, z + k)
                        if im[neighbor_index] > threshold:
                            connected_points.add(neighbor_index)
    #print("Points with values above threshold and 6-connectivity:", connected_points)
    return list(connected_points)



if __name__ == '__main__':

    data_path = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/raw/annotated_data_bii/PD190/og_ct.nii'
    aorta_path = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/raw/annotated_data_bii/PD190/aorta_mask.nii'
    markup_path = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/raw/annotated_data_bii/PD190/bifurcation.json'

    """ 
    losses = map2Dto3D(pred_labels_path=path, ct_images_path=data_path)
    print(losses)
    average = sum(losses.values()) / len(losses)
    print("Average:", average)
    """

    

    bifurcation = utils.get_3Dcoor_from_markup(markup_path, data_path)

    get_LM(data_path, aorta_path, bifurcation)
    







