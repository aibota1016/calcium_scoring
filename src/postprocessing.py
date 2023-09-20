from .utils import read_label_txt



def extract_LM(predicted_label_txt, ):
    """a method to map predicted bbox labels to original slice"""
    """
    if label txt file is not empty:
        read corresponding ct image, fix direction
        take corresponding slice, map the bounding box coordinates
        calculate nearest neighbor distance from aorta center to bifurcation center, draw a line
        
    """