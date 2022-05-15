import pandas as pd 
import argparse
import sys
from code.utils import colorizar

DATA_PATH = "data/train.csv"
PERCENT = 0.05

assert PERCENT > 0 and PERCENT < 1

def check_params(arg=None):
    parse = argparse.ArgumentParser(description='Data spliter for the model')

    parse.add_argument('--createppl', help='create the LP problem to solve', 
					   required=False, action='store_true', default=True)
    
    parse.add_argument('--createdata', help='create the dataset with the LP solution', 
					   required=False, action='store_true', default=False)

    returns = parse.parse_args(arg)
    B1    = bool(returns.createppl)
    B2    = bool(returns.createdata)

    return 1 if B1 and not B2 else 2

def calculateGroups():
    data = pd.read_csv(DATA_PATH)[["reply_to", "comment_id"]]

    cid = list(set(data["comment_id"]))

    mark = {}
    groups, currG = [], []
    for i in cid:
        if mark.get(i) is None:
            
            foundG = -1
            rep = data.query(f"comment_id == {i}")["reply_to"].to_list()[0]
            currG = [i]

            while rep != i:
                i = rep 
                rep = data.query(f"comment_id == {i}")["reply_to"].to_list()[0]
                
                if mark.get(i) is not None:
                    foundG = mark[i]
                    break
                
                currG.append(i)
            
            if foundG == -1:
                foundG = len(groups)
                groups.append([i])
            else:
                groups[ foundG ] += currG

            for v in currG:
                mark.update({v:foundG})

    return groups

def calcWeights(L:list):
    data = pd.read_csv(DATA_PATH)[["stereotype", "comment_id"]]

    v_pos, v_neg = 0, 0
    for ide in L:
        labels = data.query(f"comment_id == {ide}")["stereotype"].to_list()
        posi   = sum([ int (h) for h in labels])
        v_pos += posi
        v_neg += len(labels) - posi
    
    return v_pos, v_neg

def calcConstraingLimit():
    data = pd.read_csv(DATA_PATH)

    pos = data.query("stereotype == 1")["stereotype"].to_list()
    pos = sum([int(v) for v in pos])

    neg = len(data) - pos

    print ("Positive samples:", pos, "{:.4}%".format(pos/len(data)))
    print ("Negative samples:", neg, "{:.4}%".format(neg/len(data)))
    print (f"Ideal {PERCENT*100.0}%: {int(len(data) * PERCENT)}")

    return int(pos * PERCENT), int(neg * PERCENT)

if __name__ == '__main__':
    ck = check_params(arg=sys.argv[1:])
    if ck == 0:
        exit(0)
    
    if ck == 1:
        groupsId = calculateGroups()
        groupsWe = list(map(lambda p: calcWeights(p), groupsId))
        pos , neg = calcConstraingLimit() 

        with open("data/prog.lp", 'w') as file:
            file.write("Maximize\n\tobj:")

            for i, pair in enumerate(groupsWe):
                file.write(f" { '+ ' if i > 0 else ' ' }x{i+1}")
            
            file.write("\nSubject To\n\tc1:")

            for i, pair in enumerate(groupsWe):
                if pair[0] != 0:
                    file.write(f" { '+ ' if i > 0 and pair[0] > 0 else  '' }{abs(pair[0])} x{i+1}")
            
            file.write(f" <= {pos}\n\tc2:")

            for i, pair in enumerate(groupsWe):
                if pair[1] != 0:
                    file.write(f" { '+ ' if i > 0 and pair[1] > 0 else  '' }{abs(pair[1])} x{i+1}")
            
            file.write(f" <= {neg}\nBinary\n\t")
            for i in range(len(groupsWe)):
                file.write(f" x{i+1}")
            
            file.write("\nEnd\n")

        with open("data/prog_vars.txt", 'w') as file:
            for i, l in enumerate(groupsId):
                file.write(f"x{i+1} {' '.join([str(s) for s in l])}\n")

        # run glpsol --cpxlp data/prog.lp -w data/output.txt 
    else:
        var = {}
        TRAIN, EVAL = [], []

        with open("data/prog_vars.txt", 'r') as file:
            for line in file.readlines():
                line = line.split(' ')
                varId = int(line[0].replace('x', ''))
                var.update({varId: [int(v) for v in line[1:]]})

        with open("data/output.txt", 'r') as file:
            for line in file.readlines():
                line = line.split(' ')
                
                if line[0] != 'j':
                    continue
                
                varNumber, value = int(line[1]), int(line[2])

                if value > 0: # goes to eval
                    EVAL += var[varNumber] 
                else:
                    TRAIN += var[varNumber] 
        
        data = pd.read_csv(DATA_PATH)

        data_train = data.query(f'comment_id in {TRAIN}')
        data_eval = data.query(f'comment_id in {EVAL}')
        
        data_train.to_csv("data/myTrain.csv", index=None)
        data_eval.to_csv("data/myEval.csv", index=None)

        print ("Data saved in", colorizar("data/myTrain.csv"), colorizar("data/myEval.csv"))
        print ("Generated {:.4}% {}".format(len(data_eval)/len(data) * 100, len(data_eval)))