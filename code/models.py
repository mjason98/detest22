import numpy as np 
import pandas as pd
import math, os, random, torch, copy
from .params import PARAMETERS
from .utils import MyBar, colorizar

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

LANGUAGE = PARAMETERS["default_language"]
TRANS_NAME = PARAMETERS["transformers_by_language"][LANGUAGE]

def setSeed(my_seed:int):
    torch.manual_seed(my_seed)
    np.random.seed(my_seed)
    random.seed(my_seed)

# function that creates the transformer and tokenizer for later uses
def make_trans_pretrained_model(model_only=False):
    '''
        This function return (tokenizer, model)
    '''
    tokenizer, model = None, None
    
    tokenizer = AutoTokenizer.from_pretrained(TRANS_NAME)
    model = AutoModel.from_pretrained(TRANS_NAME)

    if model_only:
        return model 
    else:
        return tokenizer, model

# The encoder last layers
class Encod_Last_Layers(torch.nn.Module):
    def __init__(self, vec_size):
        super(Encod_Last_Layers, self).__init__()
        
        self.__MHAtttentionLayer  = torch.nn.TransformerEncoderLayer(vec_size, nhead=3, batch_first=True)
        self.MHA = torch.nn.TransformerEncoder(self.__MHAtttentionLayer, num_layers=1)
        self.vec_size = vec_size
        # Classification
        self.Task1    = torch.nn.Linear(vec_size, 2)

    def forward(self, X, V1, V2):
        batchS = X.shape[0]
        x_ = torch.concat([ X.view(batchS, 1, self.vec_size) , V1.view(batchS, -1, self.vec_size), V2.view(batchS, -1, self.vec_size)], axis=1)
        x_ = self.MHA(x_)
        x_ = self.Task1(x_[:,-1,:])
        return x_

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)

# The encoder used in this work
class Encoder_Model(torch.nn.Module):
    def __init__(self):
        super(Encoder_Model, self).__init__()
        self.criterion1 = torch.nn.CrossEntropyLoss( torch.Tensor(PARAMETERS["training_params_by_language"][LANGUAGE]["training_weights"]) )

        self.max_length = PARAMETERS["training_params_by_language"][LANGUAGE]["max_lenght"]
        self.tok, self.bert = make_trans_pretrained_model()

        self.encoder_last_layer = Encod_Last_Layers(PARAMETERS["training_params_by_language"][LANGUAGE]["transformer_embedding_size"])

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.to(device=self.device)
        
    def forward(self, X, V1, V2, return_vec=False):
        ids   = self.tok(X, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length).to(device=self.device)
        out   = self.bert(**ids)
        vects = out[0][:,-1]

        if return_vec:
            return vects
        
        return self.encoder_last_layer(vects, V1, V2)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

    def save(self, path):
        torch.save(self.state_dict(), path) 
    
    def makeOptimizer(self):
        pars = [{'params':self.encoder_last_layer.parameters()}]

        lr = PARAMETERS["training_params_by_language"][LANGUAGE]["lr"]
        lr_factor = PARAMETERS["training_params_by_language"][LANGUAGE]["lr_factor"]

        for l in self.bert.encoder.layer:
            lr *= lr_factor
            D = {'params':l.parameters(), 'lr':lr}
            pars.append(D)
        try:
            lr *= lr_factor
            D = {'params':self.bert.pooler.parameters(), 'lr':lr}
            pars.append(D)
        except:
            print('#Warning: Pooler layer not found')

        if PARAMETERS["training_params_by_language"][LANGUAGE]["encoder_optimizer"] == 'adam':
            
            return torch.optim.Adam(pars, lr=lr, weight_decay=PARAMETERS["training_params_by_language"][LANGUAGE]["encoder_decay"])
        
        elif PARAMETERS["training_params_by_language"][LANGUAGE]["encoder_optimizer"] == 'rms':
            
            return torch.optim.RMSprop(pars, lr=lr, weight_decay=PARAMETERS["training_params_by_language"][LANGUAGE]["encoder_decay"])

def trainModels(model, Data_loader, evalData_loader=None, nameu='encoder', optim=None):
    if optim is None:
        optim = torch.optim.Adam(model.parameters(), lr=PARAMETERS["training_params_by_language"][LANGUAGE]["lr"])
    model.train()

    epochs = PARAMETERS["training_params_by_language"][LANGUAGE]["epochs"]
    
    changeFreq = PARAMETERS["training_params_by_language"][LANGUAGE]["target_frequency"]
    targetNet = copy.deepcopy(model).to(device=model.device)
    targetNet.eval()
    
    best_acc, borad_train, board_eval = 0, [], []
    for e in range(epochs):

        if e == 0 or e%changeFreq == 0:
            targetNet.load_state_dict(model.state_dict())
            targetNet.eval()

        bar = MyBar('Epoch '+str(e+1)+' '*(int(math.log10(epochs)+1) - int(math.log10(e+1)+1)) , 
                    max=len(Data_loader)+(len(evalData_loader) if evalData_loader is not None else 0))
       
        total_loss, total_acc, dl = 0., 0., 0
        for data in Data_loader:
            optim.zero_grad()

            v1 = [y for x in data['v1'] for y in x]
            v2 = [y for x in data['v2'] for y in x]

            with torch.no_grad():
                v1 = targetNet(v1, None, None, return_vec=True).detach()
                v2 = targetNet(v2, None, None, return_vec=True).detach()

            y_hat = model(data['x'], v1, v2)
            y1    = data['y'].to(device=model.device).flatten()
            
            del v1 
            del v2 
            
            try:
                loss = model.criterion1(y_hat, y1)
            except:
                # size 1
                y_hat = y_hat.view(1,-1)
                loss  = model.criterion1(y_hat, y1)
            
            loss.backward()
            optim.step()

            with torch.no_grad():
                total_loss += loss.item() * y1.shape[0]
                total_acc  += (y1 == y_hat.argmax(dim=-1).flatten()).sum().item()
                dl += y1.shape[0]
            bar.next(total_loss/dl)
        
        borad_train.append(total_acc/dl)
        
        # Evaluate the model
        if evalData_loader is not None:
            total_loss, total_acc, dl= 0,0,0
            
            with torch.no_grad():
                for data in evalData_loader:

                    v1 = [y for x in data['v1'] for y in x]
                    v2 = [y for x in data['v2'] for y in x]

                    v1 = model(v1, None, None, return_vec=True).detach()
                    v2 = model(v2, None, None, return_vec=True).detach()

                    y_hat = model(data['x'], v1, v2)
                    y1    = data['y'].to(device=model.device).flatten()
                    loss = model.criterion1(y_hat, y1)
                    
                    total_loss += loss.item() * y1.shape[0]
                    total_acc += (y1 == y_hat.argmax(dim=-1)).sum().item()
                    dl += y1.shape[0]
                    bar.next()
            
            if best_acc < total_acc:
                best_acc = total_acc
                model.save(os.path.join('pts', nameu+'.pt'))
            
            board_eval.append(total_acc/dl)

        bar.finish()
        del bar
    
    return borad_train, board_eval


def predict(model, dataloader, filename, oldfile, drop=['reply_to', 'sentence']):
    filename = os.path.join('data', filename+'.csv')
    model.eval()

    pred = []
    dataO = pd.read_csv(oldfile).drop(drop, axis=1)
    newColumn = list(dataO.columns) + ['stereotype']

    bar = MyBar('eval', max=len(dataloader))
    with torch.no_grad():
        for data in dataloader:

            v1 = [y for x in data['v1'] for y in x]
            v2 = [y for x in data['v2'] for y in x]

            v1 = model(v1, None, None, return_vec=True).detach()
            v2 = model(v2, None, None, return_vec=True).detach()

            y_hat = model(data['x'], v1, v2).argmax(dim=-1).squeeze().cpu().numpy()
            pred.append(y_hat)            

            bar.next()
    bar.finish()
    
    pred = pd.Series(np.concatenate(pred))
    dataO = pd.concat([dataO, pred], axis=1)

    dataO.to_csv(filename, index=None, header=newColumn)
    print("# predictions saved in", colorizar(filename))



class RawDataset(Dataset):
    def __init__(self, csv_file, comentId='comment_id', sentence='sentence', classHeader='stereotype', replay="reply_to"):
        self.data_frame = pd.read_csv(csv_file)
        self.sentenceH  = sentence
        self.comentIdH = comentId
        self.classH = classHeader
        self.replyToH = replay
        self.window  = 3
        self.max_length = len(self.data_frame)

    def __len__(self):
        return self.max_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        commentid = self.data_frame.loc[idx, self.comentIdH]
        replayto  = self.data_frame.loc[idx, self.replyToH]
        sentence  = self.data_frame.loc[idx, self.sentenceH]

        rangeSame  = self.data_frame[max(idx - self.window, 0) : idx                                   ].query(f"{self.comentIdH} == {commentid}")[self.sentenceH].tolist()
        rangeSame += self.data_frame[idx+1                     : min(idx+self.window, self.max_length) ].query(f"{self.comentIdH} == {commentid}")[self.sentenceH].tolist()
        rangeSame  = [""]*(self.window*2-1 - len(rangeSame)) + rangeSame

        rangeReplay = self.data_frame.query(f"{self.comentIdH} == {replayto}")[self.sentenceH].tolist()

        if len(rangeReplay) > self.window*2-1:
            rangeReplay = rangeReplay[-self.window*2+1:]
        elif len(rangeReplay) < self.window*2-1:
            rangeReplay = [""]*(self.window*2-1 - len(rangeReplay)) + rangeReplay
        
        try:
            classValue =  int(self.data_frame.loc[idx, self.classH])
        except:
            classValue = -1

        sample = {'x': sentence, 'v1': rangeSame, 'v2':rangeReplay, 'y':classValue}
        return sample

def makeDataSet(csv_path:str, shuffle=True):
    batch   = PARAMETERS["training_params_by_language"][LANGUAGE]["batch"]
    id_h    = PARAMETERS["dataset_info"]["comentId_header"]
    text_h  = PARAMETERS["dataset_info"]["sentence_header"]
    class_h = PARAMETERS["dataset_info"]["class_header"]
    repl_h  = PARAMETERS["dataset_info"]["replay_header"]
    WORKS   = PARAMETERS["workers"]

    data   =  RawDataset(csv_path, comentId=id_h, sentence=text_h, classHeader=class_h, replay=repl_h)
    loader =  DataLoader(data, batch_size=batch, shuffle=shuffle, num_workers=WORKS, drop_last=False)
    
    return loader