import streamlit as st
import os
from werkzeug.utils import secure_filename
import predict
import utils
import postprocessing
import shutil
import json
from visualizations import viz


# Set the upload folder and allowed extensions
UPLOAD_FOLDER = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/src/predictions'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'nii', 'nii.gz'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Function to perform object detection
def detect_bifurcation(image_path, model_path='/Users/aibotasanatbek/Documents/projects/calcium_scoring/experiments/tuned/train/weights/best.pt'):
    # Placeholder function that returns a dummy result
    predict.predict_bifurcation(trained_model=model_path, ct_path=image_path)
    predicted_images = predict.process_output(image_path)
    return predicted_images

def perform_aorta_segmentation(image_name):
    inference_base = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/src/aorta_segmentation/prediction_testing'
    seg_path = predict.get_segmentation(image_name, inference_base)
    return seg_path


def show_output(image_path, filename, options):
    predicted_images = detect_bifurcation(image_path)
    if options.get("Option 1"):
        # Display the predicted images and results
        st.header("Bifurcation Prediction Results")
        imgs = st.columns(len(predicted_images))
        for i, image in enumerate(predicted_images):
            imgs[i].image(image, caption=f'{image.split("_")[-1].split(".")[0]}th slice', use_column_width=True)
        st.write("The predicted images are shown above.")
    
    if options.get("Option 2"):
        # Add download button to download generated json file
        st.header("Download Result File")
        st.markdown("""
            Click the button below to download the json file with 3D bbox labels that can be imported to 3D Slicer Software.
        """)
        existing_json_filepath = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/src/predictions/predict/pred.json'
        if os.path.exists(existing_json_filepath):
            with open(existing_json_filepath, 'r') as json_file:
                json_content = json_file.read()
        else:
            # If the file doesn't exist, provide a default value or handle accordingly
            json_content = None 
        st.download_button(
            label="Download Bbox Prediction Json File",
            data=json_content,
            file_name='bifurcation.json',
            key='json_file_button'
        )
    
    aorta_file_path = perform_aorta_segmentation(filename)
    print(aorta_file_path)
    
    #aorta_file_path = '/Users/aibotasanatbek/Desktop/data/PD065/aorta_mask.nii'
    
    if options.get("Option 3"):
        viz.plot_masks(utils.read_nifti_image(image_path), utils.read_nifti_image(aorta_file_path), save_path='/Users/aibotasanatbek/Documents/projects/calcium_scoring/src/predictions/predict/aorta_mask.png')
        st.image('/Users/aibotasanatbek/Documents/projects/calcium_scoring/src/predictions/predict/aorta_mask.png', use_column_width=True)

    st.header("LM calcium detection results")
    markup_path = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/src/predictions/predict/pred.json'
    bifurcation = utils.get_3Dcoor_from_markup(markup_path, image_path)
    connected_points = postprocessing.detect_LM_calcium(image_path, aorta_file_path, bifurcation)

    if len(connected_points) > 0:
        st.write("There is a presence of LM calcification in the given CT scan")
        st.write(f"The volume of the calcification consists of {len(connected_points)} pixels with 6-connectivity")
    else:
        st.write("There is no presence of LM calcification in the given CT scan")
    
    shutil.rmtree('/Users/aibotasanatbek/Documents/projects/calcium_scoring/src/predictions')

    

def main():
    st.title("LM Calcium Detection Web App")
    #st.set_page_config(page_title="LM Calcium Detection Web App", page_icon="🤖")

    # Set favicon
    st.markdown("""
        <link rel="shortcut icon" type="image/x-icon" href="data:image/x-icon;,">
    """, unsafe_allow_html=True)

    # Description Section
    st.header("Description")
    st.write("""
        Welcome to the Calcium Detection Web App for my final year project! \n
        This app detects the presence of calcification in Left Main coronary artery from Non-Contrast CT scan images. \n
        Here are the approaches taken: \n
            1. Bifurcation point detection \n
            2. Aorta segmentation \n
            3. Calcification detection in Left Main artery \n
    """)

    # Set smaller font size for the description section
    st.markdown('<style>h2{font-size: 18px !important;}</style>', unsafe_allow_html=True)


    # Toggle List Section
    #with st.expander("Click here for more explanation about the LM calcium detection method"):
    #    st.write("This content is hidden by default. Click the toggle to show or hide.")

    # Options Section
    st.header("Results Display Options")

    # Checkboxes
    col1, col2 = st.columns(2)
    option1 = col1.checkbox("Display bifurcation point detection results")
    option2 = col2.checkbox("Display aorta segmentation results")
    option3 = st.checkbox("Generate json file with bifurcation 3D bbox (for importing to 3D Slicer software)")


    # Upload Section
    st.header("Upload NIfTI File")
    uploaded_file = st.file_uploader("Choose a CT Scan Image in NIfTI file format (.nii)", type=['nii', 'nii.gz'])

    if uploaded_file is not None:
        filename = secure_filename(uploaded_file.name)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Get selected options
        options = {
            "Option 1": option1,
            "Option 2": option2,
            "Option 3": option3,
            #"Selected Option": selected_option
        }

        

        # Display output
        show_output(filepath, filename, options)


if __name__ == '__main__':
    main()
    #image_path='/Users/aibotasanatbek/Desktop/data/PD065/og_ct.nii'
    #aorta_file_path = '/Users/aibotasanatbek/Desktop/data/PD065/aorta_mask.nii'
    #aorta_file_path = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/src/Documents/projects/calcium_scoring/src/predictions/og_ct/og_ct_seg.nii.gz'

