# Dependencies
import cv2
import numpy as np
import pandas as pd

import os
import glob
import pickle
import random
from tqdm import tqdm

import pydicom
from pydicom.data import get_testdata_files

import seaborn as sns
import matplotlib.pyplot as plt


class ImagePreprocess(object):
    """
    Class for preparing images
    """
    def __init__(self, resize, channels, norm_type):
        
        self.resize = resize
        self.channels = channels
        self.norm_type = norm_type
     
    def __rescale(self, image):
        """
        Resize and rescale the image
        """
        
        image = cv2.resize(image, (self.resize, self.resize))
        return image.reshape((self.resize, self.resize, self.channels))
        
    def __normalize(self, image):
        """
        Channel-wise normalization
        """
        
        if self.norm_type == 'min-max':
            return np.float32((image - np.min(image))/(0.00001+(np.max(image) - np.min(image))))
        
        elif self.norm_type == 'mean-var':
            return np.float32((image - np.mean(image))/((np.std(image)+0.00001)))
        
        else:
            raise ValueError("Un-identified parameter value, ntype : {}".format(ntype))
    
    def preproc_image(self, image):
        """
        Pre-process the image
        """
        normalized_img = self.__normalize(image)
        rescaled_img = self.__rescale(normalized_img)
            
        return rescaled_img 
        

class GetData(ImagePreprocess):
    """
    Class for putting together the training data 
    """
    
    def __init__(self, exam_paths, df_path, **kwargs):
        
        self.exam_paths = exam_paths 
        self.df_path = df_path
        
        super().__init__(**kwargs)
        
        
    def __get_DCMPaths(self, exam_path):
        """
        Get all examination paths
        """
        return glob.glob(exam_path + '/*')
        
        
    def __get_ExamLabel(self, exam_number):
        """
        Get EDSS exam label
        """
        df = pd.read_excel(self.df_path, sheet_name='Feuil1')#read_xls(self.df_path)
        return df[df['Sequence_id'] == exam_number]['EDSS'].values[0]
        
    
    def __get_3DImage(self, exam_DCMPaths):
        """
        Open dcm files, read image, preprocess them
        """
        no_good = ['/home/alex/Dataset 1/26108/1.3.12.2.1107.5.2.18.41433.2015031912480869998469389.dcm']
        image_list = []

        for path in exam_DCMPaths:
            if path in no_good:
                continue
            dataset = pydicom.dcmread(path)                  # Read dcm file
            image = dataset.pixel_array                      # Get image from dcm file
            image_list.append(self.preproc_image(image))  # pre-process the image
           
        # Get exam EDSS label
        exam_number = int(path.split('/')[-2])
        label = self.__get_ExamLabel(exam_number)

        return np.array(image_list), label
        
        
    def create_dataset(self):
        """
        Constructs the dataset
        """
        
        data_MRIImages = []
        data_MRILables = []
        
        for exam_path in tqdm(self.exam_paths):

            exam_DCMPaths = self.__get_DCMPaths(exam_path)
            mri_3DImage, label = self.__get_3DImage(exam_DCMPaths)

            data_MRIImages.append(mri_3DImage)
            data_MRILables.append(label)
            
        return data_MRIImages, data_MRILables


if __name__ == "__main__":

    base_path = '/home/alex/Dataset 1/'
    remove = ['Dataset - 1.xlsx']
    exam_paths = glob.glob(base_path + '/*')
    exam_paths = [i for i in exam_paths if i.split('/')[-1] not in remove]
    resize = 299
    channels = 1
    norm_type = 'min-max'

    df_path = base_path + 'Dataset - 1.xlsx'

    g = GetData(exam_paths, df_path, resize=resize, channels=channels, norm_type=norm_type)

    data_MRIImages, data_MRILabels  = g.create_dataset()

    with open('data/data_MRIImages.pkl', 'wb') as f:
        pickle.dump(data_MRIImages, f)
    with open('data/data_MRILabels.pkl', 'wb') as f:
        pickle.dump(data_MRILabels, f)
