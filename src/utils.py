## Dependencies

import os
import glob
import random

import cv2
import numpy as np
import pandas as pd

import pydicom
from pydicom.data import get_testdata_files


def create_dir(dir_path):

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print("Model save directory doesn't exist. Created directory!")
    else:
        print("Model save directory already exists")


def get_PatientInfo(database_path, split=0.9, shuffle=True, seed=100):

    """
    Get patient id and EDSS score from dataframe
    """

    df_path =  database_path + '/Dataset - 1.xlsx'
    df = pd.read_excel(df_path, sheet_name='Feuil1')

    edss = df['EDSS'].tolist()
    p_id = df['Sequence_id'].tolist()

    patient_InfoDatabase = [(p_id[i], edss[i]) for i in range(df.shape[0])]
    
    if shuffle:
            random.seed(seed)
            random.shuffle(patient_InfoDatabase)
            
    train_patient_information = patient_InfoDatabase[:int(split*len(patient_InfoDatabase))]
    valid_patient_information = patient_InfoDatabase[int(split*len(patient_InfoDatabase)):]

    return train_patient_information, valid_patient_information


class ImagePreprocess(object):
    """
    Class for preparing images
    """
    def __init__(self, resize, normalization, channels=1):

        self.channels = channels
        self.image_resize = resize
        self.normalization = normalization

    def __rescale(self, image):
        """
        Resize and rescale the image
        """

        image = cv2.resize(image, (self.image_resize, self.image_resize))
        return image.reshape(( self.channels, self.image_resize, self.image_resize))

    def __normalize(self, image):
        """
        Channel-wise normalization
        """

        if self.normalization == 'min-max':
            return np.float32((image - np.min(image))/(0.00001+(np.max(image) - np.min(image))))

        elif self.normalization == 'mean-var':
            return np.float32((image - np.mean(image))/((np.std(image)+0.00001)))

        else:
            raise ValueError("Un-identified parameter value, ntype : {}".format(ntype))

    def preproc_image(self, image):
        """
        Pre-process the image by applying normalization and 
        resizing
        """
        normalized_img = self.__normalize(image)
        rescaled_img = self.__rescale(normalized_img)

        return rescaled_img 


    def transform_images(self, image, transformation='original', angle=30):
        """
        Method to generate images based on the requested transfomations
        Args:
        - image             (nd.array)  : input image array
        - transformation    (str)       : image transformation to be effectuated
        - angle 		(int)	    : rotation angle if transformation is a rotation
        Returns:
        - trans_image       (nd.array)  : transformed image array
        """
        
        def rotateImage(image, angle):
            """
            Function to rotate an image at its center
            """
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
            return result

        # Image transformations
        if transformation == 'original':
            trans_image = image
        elif transformation == 'flip_v':
            trans_image = cv2.flip(image, 0)
        elif transformation == 'flip_h':
            trans_image = cv2.flip(image, 1)
        elif transformation == 'flip_vh':
            trans_image = cv2.flip(image, -1)
        elif transformation == 'rot_c':
            trans_image = rotateImage(image, -angle)
        elif transformation == 'rot_ac':
            trans_image = rotateImage(image, angle)
        else:
            raise ValueError("In valid transformation value passed : {}".format(transformation))

        return trans_image


