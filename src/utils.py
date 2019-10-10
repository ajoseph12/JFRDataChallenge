## Dependencies

import glob

import cv2

import pydicom
from pydicom.data import get_testdata_files


class SEP_generator(ImagePreprocess):
    """
    Generator Class
    """
    
    def __init__(self, patient_InfoDatabase, base_DatabasePath, **kwargs):
        """
        Args:
        - patient_information	(list) : list of tuples : (patient_id, patient_edss_score) 

        """
        self.resize 									= kwargs['resize']
        self.patient_InfoDatabase						= patient_information
        self.base_DatabasePath                          = base_DatabasePath
        self.nogood_DCMPaths 							= ['/home/alex/Dataset 1/26108/1.3.12.2.1107.5.2.18.41433.2015031912480869998469389.dcm']

        super().__init__(**kwargs)

    def __get_DCMFilePaths(self, patient_information, base_DatabasePath='/home/alex/Dataset 1'):
        """
        Internal method to get path towards patient dcm files
        """
        patient_id, patient_label = patient_information
        patient_path = base_DatabasePath + '/' + str(patient_id) + '/*.dcm'

        patient_dcm_FilePaths = glob.glob(patient_path)

        return patient_dcm_FilePaths, patient_label


    def __extract_DCMImage(self, dcm_path):

        dataset = pydicom.dcmread(dcm_path)       	# Read dcm file
        image = dataset.pixel_array					# Exctact image from dcm files 

        return image


    def generator(self, max_slices=70, dark_matter=0.7):

        """
        Args:
        - max_slices	(int) 		: number of allowed slices per patient
        - dark_matter 	(int) 		: ratio of dark matter accepted in an image

        Yields:
        - image_3D		(nd.array)	:
        - patient_label	(float)		:
        """

        itr = 1
        limit = len(self.patient_InfoDatabase)

        while True:
            
            patient_information = self.patient_InfoDatabase[itr]
            print("Iteration : {}".format(itr, patient_information))
            # Get list of patient dcm file paths and respective scores
            # [[p1_01.dmc, ..  p1_99.dcm], p1_label]
            patient_dcm_FilePaths, patient_label 	= self.__get_DCMFilePaths(patient_information, base_DatabasePath)

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

            yield image_3D[relevant_idx], patient_label

            itr += 1

            if itr == limit:
                itr == 0


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


