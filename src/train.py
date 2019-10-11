
import pandas as pd


def training(database_path,
                model_path,

                resize,
                channels,
                normalization,
                lr,
                epochs,
                activation,
                classes=None,
                weights=None,
                dataset='sep',
                trainvalsplit =
                ):


    

    # channels = channels
    # resize = resize
    # normalization = 'min-max'


    if dataset == 'sep':

        df_path =  dataset + 'Dataset - 1.xlsx'
        df = pd.read_excel(file_path, sheet_name='Feuil1')

        edss = df['EDSS'].tolist()
        p_id = df['Sequence_id'].tolist()

        patient_information = [(p_id[i], edss[i]) for i in range(df.shape[0])]
        train_patient_information = patient_information[:int(0.9*len(patient_information))]
        valid_patient_information = patient_information[int(0.9*len(patient_information)):]


        train
        base_DatabasePath = '/home/alex/Dataset 1'


        train_inst = utils.SEP_generator(base_DatabasePath, 
                                                channels=channels,
                                                resize=resize,
                                                normalization=normalization)

        train_generator = generator_inst.generator(patient_InfoDatabase)



    for m, item in enumerate(train_generator):
        image_3D, label = torch.tensor(item[0], device=device).float(), torch.tensor(item[1], device=device).float()
        print(image_3D.shape)
        output = net1(image_3D, 1)
        loss = criterion(output, label)
        
        loss.backward()
        optimizer.step()





def parse_command_line():
    """
    Function to parse command line arguments
    """

    version = "%prog 1.0"
    usage   = "usage %prog [options]"
    parser  = OptionParser(usage=usage, version=version)

    parser.add_option("-d", "--dataset", dest="dataset", help="Which dataset to use (mura or patch)")
    parser.add_option("-f", "--filepath", dest="filepath", help="Path into which the weights should be stored")
    parser.add_option("-rs", "--resize", dest="resize", help="Image to be re-size to", type=int, default=299)
    #parser.add_option("-b", "--batch_size", dest="batch_size", help="Batch size", type=int, default=1)    
    parser.add_option("-c", "--channels", dest="channels", help="{1:'grayscale', 3:'rgb', 4:'rgba'}", type=int, default=1)
    parser.add_option("-n","--norm", dest="norm", help="Type of normalization to use", type=str, default=None)
    parser.add_option("-a","--acti", dest="acti", help="Type of activation to use at the last layer of VGG", type=str, default='softmax')
    parser.add_option('--weights', dest="weights", help='Initialize the model with weights from a file.', default=None)
    parser.add_option("--classes", dest='classes', type=int, help="number of classes the input dataset has", default=2)
    parser.add_option("--epochs", dest="epochs", help="Number of training epochs", type=int)
    parser.add_option("--lr", dest="learnrate", help="Learning rate during training", type=float)

    return parser.parse_args()

if __name__ == '__main__':

training()




