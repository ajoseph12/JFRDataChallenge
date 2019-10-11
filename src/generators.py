import glob
import random

import cv2
import numpy as np

import pydicom
from pydicom.data import get_testdata_files

from utils import ImagePreprocess


class SEPGenerator(ImagePreprocess):
    """
    Generator Class
    """
    
    def __init__(self, base_DatabasePath, **kwargs):
        """
        Args:
        - base_DatabasePath
        """
        self.resize 									= kwargs['resize']
        self.base_DatabasePath                          = base_DatabasePath
        self.nogood_DCMPaths 							= ['/home/alex/Dataset 1/26108/1.3.12.2.1107.5.2.18.41433.2015031912480869998469389.dcm']
        
        super().__init__(**kwargs)

    def __get_DCMFilePaths(self, patient_information):
        """
        Internal method to get path towards patient dcm files
        """
        patient_id, patient_label = patient_information
        patient_path = self.base_DatabasePath + '/' + str(patient_id) + '/*.dcm'

        patient_dcm_FilePaths = glob.glob(patient_path)

        return patient_dcm_FilePaths, patient_label


    def __extract_DCMImage(self, dcm_path):

        try:
            dataset = pydicom.dcmread(dcm_path)       	# Read dcm file
            image = dataset.pixel_array					# Exctact image from dcm files 
            return image

        except exception as e:
            print(e, path)


    def generator(self, patient_InfoDatabase, max_slices=70, dark_matter=0.7, shuffle=True):

        """
        Args:
        - patient_InfoDatabase	(list)      : list of tuples : (patient_id, patient_edss_score) 
        - max_slices	        (int)       : number of allowed slices per patient
        - dark_matter 	        (int)       : ratio of dark matter accepted in an image

        Yields:
        - image_3D		        (nd.array)	:
        - patient_label	        (float)		:
        """

        itr = 0
        limit = len(patient_InfoDatabase)

        while True:
            
            patient_information = patient_InfoDatabase[itr]
            #print("Iteration : {} | {}".format(itr, patient_information))
            # Get list of patient dcm file paths and respective scores
            # [[p1_01.dmc, ..  p1_99.dcm], p1_label]
            patient_dcm_FilePaths, patient_label 	= self.__get_DCMFilePaths(patient_information)

            # Select a random transformation 
            transformation = random.choice(['original', 'flip_v', 'flip_h', 'flip_vh', 'rot_c', 'rot_ac'])
            
            # Create array of zeors # (x, 1, 299, 299) --> (slices, channels, height, width)
            darkmatter_idx 	= []
            image_3D 		= np.zeros((len(patient_dcm_FilePaths), 1, self.resize, self.resize))

            # Create array with relevant images from dcm files
            for n, patient_dcm_FilePath in enumerate(patient_dcm_FilePaths):
                #print("Dcm file number : {}".format(n))
                # Some dcm files are corrupted, ignore thme
                if patient_dcm_FilePath in self.nogood_DCMPaths:
                    continue

                dcm_image 		= self.__extract_DCMImage(patient_dcm_FilePath)			# extract image from .dcm file
                preproc_image 	= self.preproc_image(dcm_image)							# preprocess image
                transform_image = self.transform_images(preproc_image, transformation)	# transform the image
                image_3D[n]		= transform_image 										# add transformed image to 3D-array
                dark_matter 	= np.sum(preproc_image == 0) / (self.resize**2)			# get amount of dark matter contained in image
                darkmatter_idx.append((dark_matter, n))

            relevant_idx = [item[1] for item in sorted(darkmatter_idx)][:max_slices]    # get relevant slices

            yield image_3D[relevant_idx], np.array(patient_label)

            itr += 1

            if itr == limit:
                itr = 0

