
import os
import random
import shutil
from pathlib import Path


    

def train_test_split(source_images_folder, source_labels_folder, destination_folder, train_ratio=0.8):
    # read files
    image_files = []
    for image_file in os.listdir(source_images_folder):
        image_files.append(image_file)
    image_files = sorted(image_files)
    # split into training, testing, validation sets
    random.seed(100)
    random.shuffle(image_files)
    split_1 = int(train_ratio * len(image_files))
    split_2 = int((train_ratio + (1-train_ratio)/2) * len(image_files))
    train_images = image_files[:split_1]
    val_images = image_files[split_1:split_2]
    test_images = image_files[split_2:]
    print(f"Number of train images: {len(train_images)}\n", 
          f"test images: {len(test_images)}\n", 
          f"validation images: {len(val_images)}")
    # write to a destination folder
    if os.path.exists(destination_folder):
        destination_folder = Path(destination_folder)
        splits = [train_images, test_images, val_images]
        for i in range(3):
            if i == 0:
                split_folder = 'train'
            elif i == 1:
                split_folder = 'test'
            else:
                split_folder = 'val'
            for image in splits[i]:
                image_path = os.path.join(source_images_folder, image) 
                destination_image_path = destination_folder / "images" / split_folder / image
                destination_image_path.parent.mkdir(parents=True, exist_ok=True)
                label_path = os.path.join(source_labels_folder, image.split('.')[0] + '.txt')
                if os.path.exists(label_path):
                    destination_label_path = destination_folder / "labels" / split_folder / image
                    destination_label_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(label_path, destination_label_path)
                shutil.copy2(image_path, destination_image_path)
    else:
        print(f'The specified destination path  doesnt exist: {destination_folder}')
 
    
    
if __name__ == '__main__':
    
    directory = os.path.abspath(__file__) # current script's absolute path
    for _ in range(3):
        directory = os.path.dirname(directory) #takes us to the project directory
        
    aorta_images_folder = os.path.join(directory, "data\\processed\\aorta\\images")
    aorta_labels_folder = os.path.join(directory, "data\\processed\\aorta\\labels")
    aorta_destination_folder = os.path.join(directory, "data\\datasets\\aorta")
    
    train_test_split(aorta_images_folder, aorta_labels_folder, aorta_destination_folder)
