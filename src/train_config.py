TRAINING_PARAMS = {
    # Hyper-parameters
    "lr"				: 3e-4,			# Learning rate during trainnig.
    "epochs"			: 5,			# Number of training epochs
    "train_iterations"	: 4000,		    # Number of steps to be taken per epoch 
    "valid_iterations"  : 43,            # Number of steps to be taken during validation

    # Image Parameters
    "normalization"		: 'min-max',                                                            # normalization to be used on input images ('min-max' or 'mean-var')
    "channels"          : 1,                                                                    # Images to be fed should have 1 or 3 channels 
    "resize"	        : 299,			                                                        # Rescale the image so the largest side is max_side
    "random_transform"	: ['original', 'flip_v', 'flip_h', 'flip_vh', 'rot_c', 'rot_ac'], 		# Randomly transform image and annotations
   

    # Model Parameters
    "classes"           : None,
    "backbone" 			: "vgg", 													# Backbone model to be used by retinanet
    "trainvalsplit"		: 0.9,			                                            # Split between training and validation set
    "model_SavePath" 	: "../data/trainings/training_1_basemodel/", 			    # Path to store snapshots of models during training
    "model_LoadWeights" : None,                                                     # Initialize the model with weights from a file 
    

    # Msc Parameters
    "dataset"           : "sep",                                                    # Dataset to train on
    "database_path"     : "/home/alex/Dataset 1"                                    # Path towards the training data
}

