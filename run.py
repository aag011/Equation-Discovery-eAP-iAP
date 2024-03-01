import argparse
import numpy as np
from sklearn.model_selection import train_test_split
# from Load_Recording_Data import LoadRecordingData, Recording  # (Use this for old data)
from load_recording_data import LoadRecordingData, Recording   # (Use this for new data)
from models import Lasso, SINDy, LassoPiecewise, SINDyPiecewise, Physics
from helper import moving_filter, mse, mae, dtw, ApplyMovingFilter


def LoadData():
    intrasTrainVal, extrasTrainVal, intrasTest, extrasTest = LoadRecordingData()
    idx_train, idx_val = train_test_split(np.arange(len(intrasTrainVal)), test_size=0.20) #, random_state=42)

    extrasTrain = extrasTrainVal[idx_train]
    intrasTrain = intrasTrainVal[idx_train]
    extrasVal = extrasTrainVal[idx_val]
    intrasVal = intrasTrainVal[idx_val]


    # dataset = LoadRecordingData()
    # # idx_train, idx_test = train_test_split(np.arange(len(intrasRaw)), test_size=0.25, random_state=42)

    # extrasTrain = dataset['8k'].extras_training2
    # intrasTrain = dataset['8k'].intras_training2
    # extrasVal = dataset['8k'].extras_val2
    # intrasVal = dataset['8k'].intras_val2
    # extrasTest = dataset['8k'].extras_unseen2
    # intrasTest = dataset['8k'].intras_unseen2

    #Smooth the data by applying moving filter on intras and extras
    windowSize = 20

    # Training set
    intrasTrain = ApplyMovingFilter(intrasTrain, windowSize)
    extrasTrain = ApplyMovingFilter(extrasTrain, windowSize)

    # Validation set
    intrasVal = ApplyMovingFilter(intrasVal, windowSize)    
    extrasVal = ApplyMovingFilter(extrasVal, windowSize)

    # Test set   
    intrasTest = ApplyMovingFilter(intrasTest, windowSize)
    extrasTest = ApplyMovingFilter(extrasTest, windowSize)

    return intrasTrain, intrasVal, intrasTest, extrasTrain, extrasVal, extrasTest

# Parse arguments
parser = argparse.ArgumentParser(description='eAP iAP Equation Discovery')
parser.add_argument('--model', type=str, default='Lasso',
                    help='model name, options: [Lasso, SINDy, LassoPiecewise, SINDyPiecewise, Physics]')
parser.add_argument('--target', type=int, default=0, help='type of predicted quantity, options: [0, 1, 2], 0 means iAP, 1 means first derivative of iAP and 2 means second derivative of iAP')
parser.add_argument('--feature', type=str, default='simple', help='type of feature set, options: [simple, complex]')
args = parser.parse_args()

intrasTrain, intrasVal, intrasTest, extrasTrain, extrasVal, extrasTest = LoadData()
modelDictionary = {'Lasso': Lasso, 'SINDy': SINDy, 'LassoPiecewise': LassoPiecewise, 
                    'SINDyPiecewise': SINDyPiecewise, 'Physics': Physics}

modelDictionary[args.model].Model().RunModel(intrasTrain, intrasVal, intrasTest, extrasTrain, extrasVal, extrasTest, 
                                        args.feature == 'simple', args.target)
