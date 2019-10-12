import glob
import random

import cv2
import numpy as np
import pandas as pd

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
        patient_path =  patient_id + '/*.dcm' #self.base_DatabasePath + '/' + str(patient_id) + '/*.dcm'

        patient_dcm_FilePaths = glob.glob(patient_path)

        return patient_dcm_FilePaths, patient_label


    def __extract_DCMImage(self, dcm_path):

        try:
            dataset = pydicom.dcmread(dcm_path)       	# Read dcm file
            image = dataset.pixel_array					# Exctact image from dcm files 
            return image

        except:
            print("Path corrupt : {}".format(dcm_path))
            return None


    def generator(self, patient_InfoDatabase, max_slices=70, dark_matter=0.7, shuffle=True, train=True):

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
            if not train:
                transformation = 'original'
            else:
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
                if dcm_image is None:                                                   # if .dcm file is corrupt
                    continue
                    
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




class CcocGenerator(object):

    """
    Creates generator object for COCO images
    """

    def __init__(self, 
                 base_path='/media/data/Coco/', 
                 resize=229, 
                 batch_size=64):
        """
        The init method called during class instantiation
        Args:
        - base_path     (str) : path towards the cooc directory 
        - resize        (int) : dimensions to which image should be resized
        - batch_size    (int) : sise of batch
        """

        self.resize         = resize
        self.base_path      = base_path
        self.batch_size     = batch_size


    def __read_df(self, dataset):
        """
        The method is used to read csv file and extract paths and labels
        Args:
        - dataset   (str)       : string indicating train or validation csv file
        Returns:
        - paths     (list)      : list of image paths
        - labels    (nd.array)  : array of labels for corresponding images
        """
        
        df = pd.read_csv(self.base_path + dataset + '2017.csv')
        paths = df['Paths'].tolist()
        labels = df.drop(['Paths'], axis=1).values
        
        return paths, labels


    def __read_image(self, image_path, channels):
        """
        The method reads an image given its path
        Args:
        - image_path    (str)       : path of the image file
        - channels      (int)       : number of channels the read image should have
        Returns:
        - image         (nd.array)  : array representing image pixels 
        """
        
        if channels == 3:
            image = cv2.imread(image_path)
        elif channels == 1:
            image = cv2.imread(image_path, 0)      
        else:
            raise ValueError("Channels value not supported : {}".format(channels))
            
        return image


    def __image_preproc(self, image):
        """
        The method performs required preprocessing on the image
        Args:
        - image     (nd.array)  : image array 
        - resize    (int)       : size to which the image ought to be resized
        Returns
        - image     (nd.array)  : image array that underwent preprocessing
        """
    
        # image resizing
        image = cv2.resize(image, (self.resize, self.resize))
        # min-max normalization 
        image = np.float32((image - np.min(image))/(0.00001+(np.max(image) - np.min(image))))
        
        return image


    def generator(self, dataset, channels=3):
        """
        Method that returns a generator object
        Args:
        - dataset       (str)       : string indicating train or validation set
        - channels      (int)       : number of channels the images have
        Yields:
        - batch_images  (nd.array)  : a batch of images of shape (batch_size, rows, cols, channesl)
        - batch_labels  (nd,array)  : a batch of labels of shape (batch_size, classes)
        """
        
        paths, labels = self.__read_df(dataset)
        steps = int(len(paths)/self.batch_size)
    
        # Intialization to keep track of generator
        itr = 0
        step = 0
            
        # Iterate through batches 
        # Used while loop instead of for loop beacause keras fit_generator
        # thows stop iteration exception with for loop
        while True: 
            
            # Get batch of image paths and their corresponding labels
            batch_paths = paths[itr:itr+self.batch_size]
            batch_labels = labels[itr:itr+self.batch_size, :]
            
            # Create batch array of zeros for image array
            batch_images = np.zeros((len(batch_paths), channels, self.resize, self.resize))
            
            for n, path in enumerate(batch_paths):
                
                temp_image = self.__read_image(path, channels)
                temp_image = self.__image_preproc(temp_image)
                
                if len(temp_image.shape) == 2:
                    temp_image = np.resize(temp_image, (1, temp_image.shape[0], 
                                                        temp_image.shape[1]))
                
                batch_images[n] = temp_image

            yield batch_images, batch_labels
            
            itr += self.batch_size
            step += 1

            if step == steps:
                itr = 0
                step = 0