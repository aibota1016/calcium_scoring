import streamlit as st
import os
from werkzeug.utils import secure_filename
import predict



# Set the upload folder and allowed extensions
UPLOAD_FOLDER = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/raw/annotated_data_bii/PD002'
ALLOWED_EXTENSIONS = {'nii', 'nii.gz'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to perform object detection (replace with your actual code)
def detect_bifurcation(image_path, options, model_path='/Users/aibotasanatbek/Desktop/FYP2/experiments/final2/final_tuned/train/weights/best.pt'):
    # Placeholder function that returns a dummy result
    predict.predict(trained_model=model_path, ct_path=image_path)
    predicted_images = predict.process_output()
    return predicted_images

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


    # Image Section
    #st.header("Sample Images")
    #st.image(["/Users/aibotasanatbek/Documents/projects/calcium_scoring/src/visualizations/aorta_mask_overlaid.png"], caption=["Image 1"], use_column_width=True)

    # Toggle List Section
    with st.expander("Click here for more explanation about the LM calcium detection method"):
        st.write("This content is hidden by default. Click the toggle to show or hide.")

    # Options Section
    st.header("Results Display Options")

    # Checkboxes
    col1, col2 = st.columns(2)
    option1 = col1.checkbox("Display bifurcation point detection results")
    option2 = col2.checkbox("Display aorta segmentation results")
    #option3 = col1.checkbox("Option 3")
    #option4 = col2.checkbox("Option 4")

    # Select Box
    #selected_option = st.selectbox("Select a bifurcation point detection threshold", ["Option A", "Option B", "Option C"])

    # Set smaller font size for the upload section
    #st.markdown('<style>h2{font-size: 18px !important;}</style>', unsafe_allow_html=True)
    

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
            #"Option 3": option3,
            #"Option 4": option4,
            #"Selected Option": selected_option
        }

        # Call your object detection model here and get the results
        # Replace the following line with your actual model inference code
        predicted_images = detect_bifurcation(filepath, options)


        # Display the predicted images and results
        st.header("Bifurcation Prediction Results")
        imgs = st.columns(3)
        for i, image in enumerate(predicted_images):
            imgs[i].image(image, caption=f'{image.split("_")[-1].split(".")[0]}th slice', use_column_width=True)
        st.write("The predicted images are shown above.")

        # Add download button for the generated file
        st.header("Download Result File")
        st.markdown("""
            Click the button below to download the json file with 3D bbox labels that can be imported to 3D Slicer Software.
        """)
        st.download_button(
            label="Download Bbox Prediction Json File",
            data='/Users/aibotasanatbek/Documents/projects/calcium_scoring/src/predictions/predict/pred.json',
            key='result_file_button'
        )

if __name__ == '__main__':
    main()


    #st.image("/Users/aibotasanatbek/Documents/projects/calcium_scoring/src/visualizations/aorta_mask_overlaid.png", caption="Image 1", width=image_width)
    #st.image("/Users/aibotasanatbek/Documents/projects/calcium_scoring/src/visualizations/bifurcation_point_bbox.png", caption="Image 2", width=image_width)

