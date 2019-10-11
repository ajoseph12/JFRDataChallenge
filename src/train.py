import random
 
import pandas as pd

import torch
import torch.optim as optim

from model import*
from utils import*
from generators import*
from train_config import TRAINING_PARAMS as config


def training(dataset,
                database_path,
                resize,
                channels,
                normalization,
                lr,
                epochs,
                train_iterations,
                valid_iterations,
                classes,
                backbone,
                trainvalsplit,
                model_SavePath,
                model_LoadWeights,
            ):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     # Check available device (GPU|CPU)
    mvcnn = MVCNN().to(device)                                                  # Instantiate network and aassign network to it
    criterion = nn.MSELoss()                                                    # Define loss
    optimizer = optim.Adam(mvcnn.parameters(), lr=0.0003)                       # Define optimizer


    if dataset == 'sep':
        
        # Get train and valid patient information 
        train_patient_information, valid_patient_information = get_PatientInfo(database_path, )

        # Create train and valid generators
        generator_inst = SEPGenerator(database_path, 
                                        channels=channels,
                                        resize=resize,
                                        normalization=normalization)
        train_generator = generator_inst.generator(train_patient_information)       
        valid_generator = generator_inst.generator(valid_patient_information)

   
        print("---------- Training has begun ----------")
        for epoch in range(epochs):
            total_TrainLoss = 0

            for t_m, t_item in enumerate(train_generator):
                # Get image and label and pass it through available device 
                image_3D, label = torch.tensor(t_item[0], device=device).float(), torch.tensor(t_item[1], device=device).float()
                output = mvcnn(image_3D, 1)                                         # Get output from network
                loss = criterion(output, label)                                     # Get loss
                loss.backward()                                                     # Back-propagate
                optimizer.step()                                                    # Update

                total_TrainLoss += loss

                if not (t_m+1)%100:
                    print("On_Going_Epoch : {} \t | Iteration : {} \t | Training Loss : {}".format(epoch+1, t_m+1, total_TrainLoss/(t_m+1)))
                    
                if t_m + 1 == train_iterations:
                    total_ValidLoss = 0

                    with torch.no_grad():
                        for v_m, v_item in enumerate(valid_generator):
                            image_3D, label = torch.tensor(v_item[0], device=device).float(), torch.tensor(v_item[1], device=device).float()
                            output = mvcnn(image_3D, 1)
                            total_ValidLoss += criterion(output, label)

                            if (v_m + 1) == valid_iterations:
                                break

                    print("Epoch : {} \t | Training Loss : {} \t | Validation Loss : {} ".format(epoch+1, total_TrainLoss/(t_m+1), total_ValidLoss/(v_m+1)) )                   
                    
                    torch.save(mvcnn, model_SavePath + config['backbone'] +'_'+ str(epoch+1) + '.pkl')
                    break
         


if __name__ == '__main__':

    training(dataset=config['dataset'],
                database_path=config['database_path'],
                resize=config['resize'],
                channels=config['channels'],
                normalization=config['normalization'],
                lr=config['lr'],
                epochs=config['epochs'],
                train_iterations=config['train_iterations'],
                valid_iterations=config['valid_iterations'],
                classes=config['classes'],
                backbone=config['backbone'],
                trainvalsplit=config['trainvalsplit'],
                model_SavePath=config['model_SavePath'],
                model_LoadWeights=config['model_LoadWeights'],
            )



# def parse_command_line():
#     """
#     Function to parse command line arguments
#     """

#     version = "%prog 1.0"
#     usage   = "usage %prog [options]"
#     parser  = OptionParser(usage=usage, version=version)

#     parser.add_option("-d", "--dataset", dest="dataset", help="Which dataset to use (mura or patch)")
#     parser.add_option("-f", "--filepath", dest="filepath", help="Path into which the weights should be stored")
#     parser.add_option("-rs", "--resize", dest="resize", help="Image to be re-size to", type=int, default=299)
#     #parser.add_option("-b", "--batch_size", dest="batch_size", help="Batch size", type=int, default=1)    
#     parser.add_option("-c", "--channels", dest="channels", help="{1:'grayscale', 3:'rgb', 4:'rgba'}", type=int, default=1)
#     parser.add_option("-n","--norm", dest="norm", help="Type of normalization to use", type=str, default=None)
#     parser.add_option("-a","--acti", dest="acti", help="Type of activation to use at the last layer of VGG", type=str, default='softmax')
#     parser.add_option('--weights', dest="weights", help='Initialize the model with weights from a file.', default=None)
#     parser.add_option("--classes", dest='classes', type=int, help="number of classes the input dataset has", default=2)
#     parser.add_option("--epochs", dest="epochs", help="Number of training epochs", type=int)
#     parser.add_option("--lr", dest="learnrate", help="Learning rate during training", type=float)

#     return parser.parse_args()






