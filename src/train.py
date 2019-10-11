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
                batch_size,
                train_iterations,
                valid_iterations,
                classes,
                backbone,
                trainvalsplit,
                model_SavePath,
                model_LoadWeights,
                use_mvcnn
            ):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     # Check available device (GPU|CPU)
    mvcnn = MVCNN().to(device)                                                  # Instantiate network and aassign network to it
    if not classes:
        criterion = nn.MSELoss()                                                # Define loss
        print("Loss used \t \t \t : MSELoss")
    else:
        criterion = nn.BCELoss ()
        print("Loss used \t \t \t : nn.BCELoss")
    optimizer = optim.Adam(mvcnn.parameters(), lr=lr)                           # Define optimizer


    if dataset == 'SEP':
        
        # Get train and valid patient information 
        train_patient_information, valid_patient_information = get_PatientInfo(database_path)

        # Create train and valid generators
        generator_inst = SEPGenerator(database_path, 
                                        channels=channels,
                                        resize=resize,
                                        normalization=normalization)
        train_generator = generator_inst.generator(train_patient_information)       
        valid_generator = generator_inst.generator(valid_patient_information, train=False)


    elif dataset == 'COCO':
        
        coco = CcocGenerator(base_path='/media/data/Coco/', resize=resize, batch_size=batch_size)

        train_generator = coco.generator(dataset='train', channels=channels)
        valid_generator = coco.generator(dataset='val', channels=channels)
   
    print("---------- Training has begun ----------")
    print("Learning rate \t \t \t : {}".format(lr))
    print("Batch size \t \t \t : {}".format(batch_size))
    print("Number of train iterations \t : {}".format(train_iterations))
    print("NUmber of valid iterations \t : {}".format(valid_iterations))
    print("Train validation split \t  \t : {}".format(trainvalsplit))
    print("Number of epochs \t \t : {}".format(epochs))
    print("Training dataset \t \t : {}".format(dataset))
    print("Loading models \t \t \t : {}".format(model_LoadWeights))
    print("Path to save models \t \t : {}".format(model_SavePath))

    create_dir(model_SavePath)

    for epoch in range(epochs):
        total_TrainLoss = 0

        for t_m, t_item in enumerate(train_generator):
            # Get image and label and pass it through available device 
            image_3D, label = torch.tensor(t_item[0], device=device).float(), torch.tensor(t_item[1], device=device).float()
            output = mvcnn(image_3D, batch_size, use_mvcnn)                     # Get output from network
            loss = criterion(output, label)                                     # Get loss
            loss.backward()                                                     # Back-propagate
            optimizer.step()                                                    # Update

            total_TrainLoss += loss

            if not (t_m+1)%100:
                print("On_Going_Epoch : {} \t | Iteration : {} \t | Training Loss : {}".format(epoch+1, t_m+1, total_TrainLoss/(t_m+1)))
                
            if (t_m+1) == train_iterations:
                total_ValidLoss = 0

                with torch.no_grad():
                    for v_m, v_item in enumerate(valid_generator):
                        image_3D, label = torch.tensor(v_item[0], device=device).float(), torch.tensor(v_item[1], device=device).float()
                        output = mvcnn(image_3D, batch_size, use_mvcnn)
                        total_ValidLoss += criterion(output, label)

                        if (v_m + 1) == valid_iterations:
                            break

                print("Epoch : {} \t | Training Loss : {} \t | Validation Loss : {} ".format(epoch+1, total_TrainLoss/(t_m+1), total_ValidLoss/(v_m+1)) )                   
                
                torch.save(mvcnn, model_SavePath + '/' + config['backbone'] +'_'+ str(epoch+1) + '.pkl')
                break
         



if __name__ == '__main__':

    training(dataset=config['dataset'],
                database_path=config['database_path'],
                resize=config['resize'],
                channels=config['channels'],
                normalization=config['normalization'],
                lr=config['lr'],
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                train_iterations=config['train_iterations'],
                valid_iterations=config['valid_iterations'],
                classes=config['classes'],
                backbone=config['backbone'],
                trainvalsplit=config['trainvalsplit'],
                model_SavePath=config['model_SavePath'],
                model_LoadWeights=config['model_LoadWeights'],
                use_mvcnn=config['use_mvcnn']
            )






