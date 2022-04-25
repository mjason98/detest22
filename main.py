from statistics import mode
import sys, argparse, os
from code.params import PARAMETERS
from code.models import setSeed, trainModels, Encoder_Model, makeDataSet
from code.utils import plotSequences

TRAIN_DATA_PATH = os.path.join("data", "myTrain.csv")
EVAL_DATA_PATH  = os.path.join("data", "myEval.csv")
TEST_DATA_PATH  = os.path.join("data", "test.csv")

def check_params(arg=None):
    global TRAIN_DATA_PATH
    global EVAL_DATA_PATH
    global PARAMETERS

    parse = argparse.ArgumentParser(description='Deep Model to solve DETEST22 Task')

    parse.add_argument('-t', dest='train_data', help='Train Data', 
                       required=False, default=TRAIN_DATA_PATH)
    parse.add_argument('-e', dest='eval_data', help='Evaluation Data', 
                       required=False, default=EVAL_DATA_PATH)
    
    parse.add_argument('--lang', dest='language', help='The language of the system', 
                       required=False, default=PARAMETERS["default_language"], choices=PARAMETERS["languages"])

    parse.add_argument('--seed', dest='my_seed', help='Random Seed', 
                       required=False, default=1234567)

    returns = parse.parse_args(arg)
    TRAIN_DATA_PATH = returns.train_data
    EVAL_DATA_PATH  = returns.eval_data
    PARAMETERS["default_language"] = returns.language

    setSeed(int(returns.my_seed))

    if not os.path.isdir("data"):
        os.mkdir("data")
    if not os.path.isdir("pts"):
        os.mkdir("pts")

def trainStereotypeClassifier():
    model = Encoder_Model()
    dataT  = makeDataSet(TRAIN_DATA_PATH)
    dataE  = makeDataSet(EVAL_DATA_PATH)

    tseq, eseq = trainModels(model, dataT, dataE, 'model', model.makeOptimizer())

    plotSequences([tseq,eseq], ['train', 'eval'])

if __name__ == "__main__":
    if check_params(arg=sys.argv[1:]) == 0:
        exit(0)
    
    trainStereotypeClassifier()