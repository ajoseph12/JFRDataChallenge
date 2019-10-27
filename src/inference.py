"""
Create .csv file with outputs from the model for corresponding patients
"""

import csv
import torch
import torch.nn as nn
import torch.optim as optim


def main(channels, resize, normalization, database_path, model_LoadWeights, csv_file):
    """
    Inferencing 
    """

    keys = {0: 0.0,
        1: 1.0,
        2: 1.5,
        3: 2.0,
        4: 2.5,
        5: 3.0,
        6: 3.5,
        7: 4.0,
        8: 4.5,
        9: 5.0,
        10: 5.5,
        11: 6.0,
        12: 6.5,
        13: 7.0,
        14: 7.5,
        15: 8.0,
        16: 8.5,
        17: 9.0}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load saved model
    mvcnn = torch.load(model_LoadWeights)

    # Get patient information
    test_patient_information = utils.get_PatientInfo(database_path, test=True)

    # Create test generator
    sep = generators.SEPGenerator(base_DatabasePath=database_path, 
                                    channels=channels,
                                    resize=resize,
                                    normalization=normalization)
    test_generator = sep.generator(test_patient_information, dataset='test')    


    # Calculate patient EDSS scores
    final_scores = []
    with torch.no_grad():
        for v_m, v_item in enumerate(test_generator):
            image_3D, p_id = torch.tensor(v_item[0], device=device).float(), v_item[1]
            if image_3D.shape[0] == 0:
                print(p_id)
                continue
            output = mvcnn(image_3D, batch_size=1, mvcnn=True)
            print(output, p_id)
            final_scores.append((p_id, output.to('cpu').detach().numpy()))
            if v_m == len(test_patient_information) - 1:
                break

    # Create and save a .csv file
    csvData = [["Sequence_id"],["EDSS"]] + list(map(lambda a : [int(a[0]), keys[np.argmax(a[1])]], (final)))
    with open(csv_file, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()


if if __name__ == "__main__":

    channels = 1
    resize = 296
    normalization = 'min-max'
    database_path = '/home/alex/Dataset3/'
    model_LoadWeights = None
    
    csv_file = 'AZmed_Unet.csv'

    
    main(channels, resize, normalization, database_path, model_LoadWeights, csv_file)