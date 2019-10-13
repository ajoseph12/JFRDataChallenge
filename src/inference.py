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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load saved model
    mvcnn = torch.load(model_LoadWeights)

    # Get patient information
    test_patient_information = utils.get_PatientInfo(database_path', test=True)

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
    csvData = [["Sequence_id","EDSS"]] + list(map(lambda a: [int(a[0]),a[1][0][0]], (final_scores)))
    with open(csv_file + '.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()


if if __name__ == "__main__":

    channels = 1
    resize = 299
    normalization = 'min-max'
    database_path = '/home/alex/Dataset3/'
    model_LoadWeights = None
    
    csv_file = None

    
    main(channels, resize, normalization, database_path, model_LoadWeights, csv_file)