import pandas as pd 
import numpy as np 
from collections import Counter
import matplotlib.pyplot as plt
import copy
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import class_weight
from torchmetrics import Accuracy
from sklearn.linear_model import LogisticRegression


def create_training_data_loaders(trainX: np.array, trainY: np.array,feature_indices:list,batch_size=1):
    X_torch = torch.Tensor(trainX[...,feature_indices])#.to(device) # transform to torch tensor
    Y_torch = torch.Tensor(trainY)#.to(device)
    train = TensorDataset(X_torch,Y_torch)
    train_data_loader = DataLoader(train, shuffle=True, batch_size = batch_size) # create your dataloader
    return train_data_loader
    
def create_prediction_data_loaders(trainX,f):
    X_torch = torch.Tensor(trainX[...,f])#.to(device) # transform to torch tensor
    train =  TensorDataset(X_torch)#CustomValLoader(X_torch,f,t)
    train_data_loader = DataLoader(train, shuffle=False, batch_size = 1) # create your dataloader
    return train_data_loader


class CNN(nn.Module):
    def __init__(self,kernel_size = 7,ensemble=1,t=0,in_filters = 1,smooth = False,drop = 0, before_or_after   = 'after', first_derivative = False):
        super(CNN, self).__init__()

        self.t = t      

        self.kernel_size = kernel_size
        self.ensembles = ensemble#*f_group
        self.in_filters = in_filters
        self.first_derivative = first_derivative
        if first_derivative:
            self.in_filters = in_filters*2
        
        self.convolution = nn.Sequential(
            nn.Conv1d(self.in_filters, self.ensembles , kernel_size=(kernel_size),padding=0,stride=1),
            nn.ReLU(inplace = True)
        )
        self.grouped_mlp = nn.Sequential(
            nn.Conv1d(self.ensembles ,2*self.ensembles ,kernel_size = 1,groups = self.ensembles )
        )

        self.drop_prob = drop
        self.before_or_after = before_or_after
        self.smooth = smooth

        
    def remove_nan_and_select_time(self,x):
        if self.t < 1: ## proportion time select
            get_non_nan_steps = torch.sum(x,dim = 1,keepdims = True)
            ind = get_non_nan_steps == get_non_nan_steps
            x = x[...,ind[0,0,...]]    
            pos = round(self.t*(x.shape[-1]))
            if self.before_or_after == "after":
                x = x[:,:,pos:]
            else:
                x = x[:,:,:pos]
        elif self.t >= 1:  ## hard time select
            if self.before_or_after == "after":
                x = x[:,:,self.t:]
            else:
                x = x[:,:,:self.t]
            get_non_nan_steps = torch.sum(x,dim = 1,keepdims = True)
            ind = get_non_nan_steps == get_non_nan_steps
            x = x[...,ind[0,0,...]]  
        return x
    
    def stochastic_max_pool(self,x, training = True):
        if training:
            if x.shape[-1] > 1:
                mask = torch.rand((x.shape[0],self.ensembles,x.shape[2]))
                mask = (mask >= self.drop_prob).float()
#                 mask = mask.repeat(1,self.control,1)
                mask = mask*2 - 1
                x_mask = x*mask
            else:
                x_mask = x
            x_mask, ind = torch.max(x_mask,dim = -1,keepdims = True)
            x = torch.gather(x,dim = -1,index = ind)
        else:
            x, ind = torch.max(x,dim = -1,keepdims = True)
        return x
    
    def weighted_average_of_ensembles(self,transformed_sequence):
        ## Weighted average of transformed sequence to be further processed
        transformed_sequence = F.conv1d(transformed_sequence, self.grouped_mlp[0].weight.detach(),self.grouped_mlp[0].bias.detach(),groups = self.ensembles)
        transformed_sequence = transformed_sequence.view(transformed_sequence.shape[0],self.ensembles,2,transformed_sequence.shape[-1])
        transformed_sequence = transformed_sequence.permute(0,2,1,3)
        ## Puts features on same scale
        transformed_sequence = (transformed_sequence - torch.mean(transformed_sequence,dim = (1),keepdims = True))#/std        
        transformed_sequence = transformed_sequence[:,1,...] - transformed_sequence[:,0,...]
        weights,ind = torch.max(torch.abs(transformed_sequence),dim = -1,keepdims =True) #/weights.shape[-1]
        avg_weight = torch.mean(weights,dim = 1,keepdims= True)
        ## Set with little seperation to .01 to limit influence
        weight_mask = (weights >= avg_weight).float() + .01
        weights= weights*weight_mask
        transformed_sequence = torch.sum(transformed_sequence*weights,dim = 1,keepdims = True)/(torch.sum(weights,dim = 1,keepdims= True) + 1e-8)
        transformed_sequence = transformed_sequence.permute(0,2,1)    
        return transformed_sequence

        

    def forward(self, x, training = True):
        x = x.permute(0,2,1)
        
        x = self.remove_nan_and_select_time(x)
        ## Average smoothing    
        x = F.avg_pool1d(x,self.smooth,1)
        
        
        ## Optional first derivatives concatenated as additional features
        if self.first_derivative:
            derive = x[:,:,1:] - x[:,:,:-1]
            derive = torch.cat([derive[:,:,0:1], derive],dim = -1)
            x = torch.cat([x,derive],dim =1)

            
        x = self.convolution(x) 

        
        transformed_sequence = copy.deepcopy(x.detach())
        
        x = self.stochastic_max_pool(x, training=training)

        x = self.grouped_mlp(x)

        

        x = x.view(x.shape[0],self.ensembles,2)
        ensemble_prediction = torch.mean(x,dim = 1,keepdims =False)
        
        
        confidence = torch.mean(x,dim = 1,keepdims = False)
        confidence = torch.softmax(confidence,dim = 1)

        transformed_sequence = self.weighted_average_of_ensembles(transformed_sequence)
    
        return ensemble_prediction, transformed_sequence, confidence
        
def train_CNN(model,train_data_loader,Y,epochs,learning_rate,balanced):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad = False)
    total_step = len(train_data_loader)

    ## gradient accumulation when batch size has to be 1 because of unequal sequence lengths due to nan values
    accum_iter = 1
    max_loss = float('inf')
    max_balanced_acc = 0
    patience = 0
    accuracy = Accuracy(average='macro',num_classes = 2).to(device)
#     if balanced == 'balanced':
#         weights = class_weight.compute_class_weight(classes = np.unique(Y), y=Y[:,0],class_weight = 'balanced')
#         weights = torch.tensor(weights.tolist(),dtype=torch.float)
#         loss_function = nn.CrossEntropyLoss(weight = weights)
#     else:
#         loss_function = nn.CrossEntropyLoss()
        
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_avg_loss = 0
        output_ = torch.zeros((0,1))
        labels_ = torch.zeros((0,1)).int()
        n = 0
        model.train()
        for i, (x, y) in enumerate(train_data_loader):  
            pred, ts, confidence = model(x,training = True)
            y_loss = torch.argmax(y,dim = -1,keepdims = False)
            loss = loss_function(pred, y_loss.long())
            ## Weight incorrect examples higher
            if (y[0,0] == 0) and (torch.argmax(pred,dim =1)[0] == 1):
                weights = confidence[0,0].detach()
                loss = loss*weights
            elif (y[0,0] == 1) and (torch.argmax(pred,dim =1)[0] == 0):
                weights = confidence[0,1].detach()
                loss = loss*weights
    
            loss.backward()
            if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_data_loader)):
                optimizer.step()
                optimizer.zero_grad()  
        model.eval()     
        for i, (x, y) in enumerate(train_data_loader):  
            pred, ts, confidence = model(x,training = False)
            y_loss = torch.argmax(y,dim = -1,keepdims = False)
            loss = loss_function(pred, y_loss.long())        
            epoch_avg_loss += loss
            pred = torch.argmax(pred,dim =-1,keepdims = True).float()
            y = torch.argmax(y,dim = -1,keepdims = True)
            output_ = torch.cat([output_,pred],axis = 0)
            labels_ = torch.cat([labels_, y.int()],axis = 0)

        epoch_avg_loss = epoch_avg_loss/(i + 1)
        output_ = (output_ > .5).float()
        balanced_acc = accuracy(output_.int(),labels_.int())
#         print("Balanced Accuracy " + str(balanced_acc))
#         print("loss " + str(epoch_avg_loss))
#         print("save " + str(epoch_avg_save_loss))
        epoch_avg_save_loss = epoch_avg_loss
        if epoch_avg_save_loss < max_loss:
            max_balanced_acc = balanced_acc
            patience = 0
            max_loss = epoch_avg_save_loss
            best_model = copy.deepcopy(model)
        else:
            patience = patience + 1
        if patience >= 2:
            break

    return best_model, max_balanced_acc
    


    
class CNN_Tree: 
    """
    Class for creating the nodes for a decision tree 
    """
    def __init__(
        self, 
        Y: np.array,
        X: np.array,
        features: list, ## List of the feature names
        use_features: list, ## If none each feature is used, 
        
        ## optional tabular features array, age, height,weight,etc.
        C_X = [], ## Np.array
        C_features = [], ## List of tabular feature names

        ## decision tree hyper parameters
        min_samples_split=20,
        max_depth=20,
        depth=0,
        node_type='root',
        rule=None,
        max_gain = .01,
        
        ## CNN hyper parameters
        standardize = True, ## this is not batch or layer normalization, but instead we normalize the data at each node
        kernel_size = 7, ## Kernel size for the CNN
        avg_smooth_size = 1, ## Smooth the Time Series Signal with moving average (Window_Size)
        n_ensembles = 1, ## Number of CNN filters to learn
        drop_pool = .25, ## Stochasistiy for the global max during CNN Training (Percentage of time steps randomly masked)
        epochs = 5, ## Number of epochs
        learning_rate = .003, ## Learning rate for the CNN
        class_weights = 'none', ## class weights for CNN and logistic regression, can be 'none' or 'balanced'
        first_derivative = False,
        
        ## time hyper parameters
        time_steps = [0], ## Time steps, either Proportion Between (0,1) or Integer > 0 and < X.shape[1]
        only_after = True, ## Whether to only look after the time step, or before as well
        weighted_distance = False, ## IF True average above threshold weighted by proportion of full sequence above threshold
        distance_function = 'mean', ## can be mean or median to calculate distance above threshold 
        
    ):

        
        self.Y = Y 
        self.X = X
        
        self.C_X  = C_X
        self.C_features = C_features
        
        self.features = features
        self.use_features = use_features if use_features else features


        
        ## CNN hyper parameters
        self.standardize = standardize ## this is not batch or layer normalization, but instead we normalize the data at each node
        self.kernel_size = kernel_size ## Kernel size for the CNN
        self.avg_smooth_size = avg_smooth_size ## Smooth the Time Series Signal with moving average (Window_Size)
        self.n_ensembles = n_ensembles ## Number of CNN filters to learn
        self.drop_pool = drop_pool ## Stochasistiy for the global max during CNN Training (Percentage of time steps randomly masked)
        self.epochs = epochs ## Number of epochs
        self.learning_rate = learning_rate ## Learning rate for the CNN
        self.class_weights = class_weights
        self.first_derivative = first_derivative
        
        ## time hyper parameters
        self.time_steps = time_steps ## Time steps, either Proportion Between (0,1) or Integer > 0 and < X.shape[1]
        self.only_after = only_after  ## Whether to only look after the time step, or before as well
        
        self.weighted_distance = weighted_distance## IF True average above threshold weighted by proportion of full sequence above threshold
        self.distance_function = distance_function ## can be mean or median to calculate distance above threshold 
        if only_after:
            self.before_or_after = ["after"]
        else:
            self.before_or_after = ["after","before"]
        
        
        self.mean = np.nanmean(self.X,axis = (0,1),keepdims = True)
        self.std = np.nanstd(self.X,axis = (0,1),keepdims = True)
        if standardize:

            self.X_Norm = (self.X - self.mean)/(self.std + 1e-9)
        else:
            self.X_Norm = self.X.copy()

        ## Decision tree Paramaters and initilizations
        self.min_samples_split = min_samples_split 
        self.max_depth = max_depth 
        self.depth = depth 
        self.min_bucket = round(self.min_samples_split/3)
        self.node_type = node_type if node_type else 'root'
        self.rule = rule if rule else ""
        self.max_gain = max_gain
        self.counts = Counter(Y.tolist())
        self.gini_impurity = self.get_GINI()
        
        
        self.prob_class_1 = np.sum(Y)/Y.shape[0]
        
        if self.prob_class_1 >= .5:
            self.yhat = 1
        else:
            self.yhat = 0
        
        ## Bucket size
        self.n = len(Y)

        self.left = None 
        self.right = None 
        
        self.best_feature = None 
        self.best_value = None 
        self.best_model = None
        self.best_time = None
        self.best_lr_boundary = None
        self.best_before_or_after = None
        self.distances = None
        self.is_cat = False

    @staticmethod
    def GINI_impurity(y1_count: int, y2_count: int) -> float:
        """
        Given the observations of a binary class calculate the GINI impurity
        """
        # Ensuring the correct types
        if y1_count is None:
            y1_count = 0
        if y2_count is None:
            y2_count = 0
        # Getting the total observations
        n = y1_count + y2_count
        # If n is 0 then we return the lowest possible gini impurity
        if n == 0:
            return 0.0
        # Getting the probability to see each of the classes
        p1 = y1_count / n
        p2 = y2_count / n
        # Calculating GINI 
#         array([0.69387755, 1.78947368])
        gini = 1 - ((p1 ** 2) +(p2 ** 2))
        # Returning the gini impurity
        return gini


    @staticmethod
    def ma(x: np.array, window: int) -> np.array:
        """
        Calculates the moving average of the given list. 
        """
        return np.convolve(x, np.ones(window), 'valid') / window

    def get_GINI(self):
        """
        Function to calculate the GINI impurity of a node 
        """
        # Getting the 0 and 1 counts
        y1_count, y2_count = self.counts.get(0, 0), self.counts.get(1, 0)
        # Getting the GINI impurity
        return self.GINI_impurity(y1_count, y2_count)
    
    def get_lr_boundary(self, model,train_data_loader):
        X = np.zeros((0,1))
        Y = np.zeros((0,1))
        sample_weights = np.zeros((0,1))
        for i, (x,y) in enumerate(train_data_loader):  
            pred, transformed_sequence, confidence = model(x,training =False)
            if (y[0,0] == 0) and (torch.argmax(pred,dim =1)[0] == 1):
                weights = confidence[0:1,0:1].detach()
            elif (y[0,0] == 1) and (torch.argmax(pred,dim =1)[0] == 0):
                weights = confidence[0:1,1:2].detach()
            else:
                weights= torch.ones((1,1)).detach()
            y = torch.argmax(y,dim = -1,keepdims = True)
            transformed_sequence = np.array(transformed_sequence.detach())
            transformed_sequence = np.squeeze(transformed_sequence,0)
            y = np.array(y.repeat(transformed_sequence.shape[0],1).detach())
            weights = np.array(weights.repeat(transformed_sequence.shape[0],1))
            X = np.concatenate((X,transformed_sequence),axis = 0)
            Y = np.concatenate((Y,y),axis = 0)
            sample_weights = np.concatenate((sample_weights,weights),axis = 0)

        clf = LogisticRegression(class_weight = self.class_weights).fit(X, Y[:,0])
        weight = clf.coef_
        bias = clf.intercept_
        pred = clf.predict(X)
        lr_boundary = -bias/weight
        unique_val = np.unique(pred)
        sens = len(np.where((pred == Y[:,0]) & (Y[:,0] == 1))[0])/len(Y[Y[:,0] == 1])
        spec = len(np.where((pred == Y[:,0]) & (Y[:,0] == 0))[0])/len(Y[Y[:,0] == 0])
        balanced_acc = (sens + spec)/2
    #         print("balanced acc " + str((sens + spec)/2))
        acc = len(np.where(pred == Y[:,0])[0])/len(Y)
    #         print("accuracy for LG = " + str(acc))
        return lr_boundary, unique_val, balanced_acc
    
    def get_distance_above_boundary(self, model,train_data_loader,lr_boundary):
        area_above_return = np.zeros((0,1))
        for i, x in enumerate(train_data_loader):  
            x = x[0]
            raw = copy.deepcopy(x)
            raw[raw != raw] = 0
            raw = np.array(raw.permute(0,2,1).detach())
            pred, transformed_sequence, confidence = model(x,training = False)
            
            ##padding so it matches raw data shape, reduced size because of kernel size.
            sequence_to_return = F.pad(transformed_sequence.permute(0,2,1),(model.kernel_size//2,model.kernel_size//2),"constant",0).permute(0,2,1)
            sequence_to_return = np.array(sequence_to_return.detach())
            
            transformed_sequence = np.array(transformed_sequence.detach())
            sequence_shift = transformed_sequence - lr_boundary
            sequence_shift = np.clip(sequence_shift,a_min = 0, a_max = None)
            
            if sequence_shift[:,np.where(sequence_shift > 0)[1],:].shape[1] > 0:
                if self.distance_function == 'median':
                    area_above = np.median(sequence_shift[:,np.where(sequence_shift > 0)[1],:],axis = 1,keepdims = False)
                else:
                    area_above = np.mean(sequence_shift[:,np.where(sequence_shift > 0)[1],:],axis = 1,keepdims = False)
            else:
                area_above = np.array([[0.]])
            if self.weighted_distance:
                l_above = sequence_shift[:,np.where(sequence_shift > 0)[1],:].shape[1]
                tot_len = sequence_shift.shape[1]
                area_above = area_above*(l_above/tot_len)

            area_above_return = np.concatenate((area_above_return, area_above))
        return area_above_return, raw, sequence_to_return
    

    def best_split(self) -> tuple:
        """
        Given the X features and Y targets calculates the best split 
        for a decision tree
        """
        # Creating a dataset for spliting
        y_val = np.expand_dims(self.Y.copy(),1)
        
        one_hot = np.int32(y_val)
        n_values = np.max(one_hot) + 1
        one_hot = np.eye(n_values)[one_hot]
        one_hot = np.squeeze(one_hot,1)        
        # Getting the GINI impurity for the base input 
        GINI_base = self.get_GINI()
        # Finding which split yields the best GINI gain 
        max_gain = self.max_gain
        for feature in self.use_features:
            
            ## Feature indices 
            f = [i for i,v in enumerate(self.features) if v in feature]
            for t in self.time_steps:
                ## All time steps up to t, or after t 
                for before_or_after in self.before_or_after:
                    
                    train_data_loader = create_training_data_loaders(self.X_Norm,one_hot,f,batch_size = 1)
                    ## Initialize and train model
                    model = CNN(kernel_size=self.kernel_size,ensemble=self.n_ensembles, ## Number of filter groups for CNN
                                t=t,
                                in_filters = len(f),
                                smooth = self.avg_smooth_size,
                                drop = self.drop_pool,
                                before_or_after = before_or_after,
                                first_derivative = self.first_derivative)#kfold_train(df_,y_val,f,to_get,t)
                    
                    
                    
                    model, CNN_balanced_acc = train_CNN(model,train_data_loader,y_val,self.epochs,self.learning_rate, self.class_weights)

                    lr_boundary, unique_val, lr_balanced_acc = self.get_lr_boundary(model,train_data_loader)
                    val_data_loader = create_prediction_data_loaders(self.X_Norm, f)
                    distances_above_lr_boundary, raw, sequence = self.get_distance_above_boundary(model, val_data_loader, lr_boundary)
                    
                    
                    ## PRUNING Defaults, if logistic regression fails to create a boundary (Predicts all 1 class, unique_val = 1)
                    ## If balanced acc of CNN better than .5
                    ## If balanced acc of the logistic regression is better than .5
                    ## Can remove, but until post pruing procedures are implemented this is a good way to prevent large trees. 
                    
                    ## Caculate GINI as normal
                    if (len(unique_val) > 1) and (CNN_balanced_acc > .5) and ( lr_balanced_acc > .5):
                        distances_above_lr_boundary = np.concatenate((distances_above_lr_boundary,y_val),axis = 1)  
                        Xdf = np.sort(distances_above_lr_boundary[:,0],axis = 0)
                        xmeans = self.ma(np.unique(Xdf), 2)
                        for value in xmeans:
                            left_counts = Counter(distances_above_lr_boundary[distances_above_lr_boundary[:,0] < value, 1])
                            right_counts = Counter(distances_above_lr_boundary[distances_above_lr_boundary[:,0] >= value,1])
                            y0_left, y1_left, y0_right, y1_right = left_counts.get(0, 0), left_counts.get(1, 0), right_counts.get(0, 0), right_counts.get(1, 0)
                            if ((y1_left + y0_left) > self.min_bucket) and ((y1_right + y0_right) > self.min_bucket):
                                gini_left = self.GINI_impurity(y0_left, y1_left)
                                gini_right = self.GINI_impurity(y0_right, y1_right)

                                n_left = y0_left + y1_left
                                n_right = y0_right + y1_right
                                w_left = n_left / (n_left + n_right)
                                w_right = n_right / (n_left + n_right)
                                # Calculating the weighted GINI impurity
                                wGINI = w_left*gini_left + w_right*gini_right
                                GINIgain = GINI_base - wGINI
                                if GINIgain > max_gain:             
                                    max_gain = GINIgain                                       
                                    self.best_time = t
                                    self.best_feature = feature
                                    self.best_value = value 
                                    self.best_model = model
                                    self.distances = distances_above_lr_boundary
                                    self.best_lr_boundary = lr_boundary
                                    self.best_before_or_after = before_or_after
                                    self.is_cat = False

                                        
        ## CNN Trees support tabular features as well (age, height, weight, gender, etc.)
        for cfeature in self.C_features:
            f = [i for i,v in enumerate(self.C_features) if v == cfeature][0]
            out = self.C_X[:,f:(f+1)]
            out = np.concatenate((out,y_val),axis = 1)  
            Xdf = np.sort(out[:,0],axis = 0)
            # Sorting the values and getting the rolling average
            xmeans = self.ma(np.unique(Xdf), 2)
            for value in xmeans:
                left_counts = Counter(out[out[:,0] < value, 1])
                right_counts = Counter(out[out[:,0] >= value,1])
                y0_left, y1_left, y0_right, y1_right = left_counts.get(0, 0), left_counts.get(1, 0), right_counts.get(0, 0), right_counts.get(1, 0)
                if ((y1_left + y0_left) > self.min_bucket) and ((y1_right + y0_right) > self.min_bucket):
                    gini_left = self.GINI_impurity(y0_left, y1_left)
                    gini_right = self.GINI_impurity(y0_right, y1_right)
                    n_left = y0_left + y1_left
                    n_right = y0_right + y1_right
                    w_left = n_left / (n_left + n_right)
                    w_right = n_right / (n_left + n_right)
                    wGINI = w_left*gini_left + w_right*gini_right
                    GINIgain = GINI_base - wGINI
                    if GINIgain > max_gain:
                        max_gain = GINIgain
                        self.best_time = None
                        self.best_feature = cfeature
                        self.best_value = value 
                        self.best_model = None
                        self.distancs = out
                        self.best_lr_boundary = None
                        self.best_before_or_after = None
                        self.is_cat = True
                                                                                  
        #if not self.is_cat:
        #    print("split TS feature " + str(self.best_feature) + " at time " + str(self.best_time) + " with gain " + str(max_gain) )
        #else:
        #    print("split Tabular feature " + str(self.best_feature) + " value " + str(self.best_value) + " with gain " + str(max_gain) ) 

    def grow_tree(self):
        """
        Recursive method to create the decision tree
        """
        # Making a df from the data 
        df = self.X.copy()
        df_C = self.C_X.copy()
        Y = self.Y.copy()

        # If there is GINI to be gained, we split further 
        unique, counts = np.unique(Y, return_counts=True)
        if (self.depth < self.max_depth) and (self.n >= self.min_samples_split) and (len(unique) > 1):
            # Getting the best split 
            self.best_split()
            if self.best_feature is not None:
                # Saving the best split to the current node 
                left_df = df[np.where(self.distances[:,0] < self.best_value)[0],...].copy()
                right_df = df[np.where(self.distances[:,0] >= self.best_value)[0],...].copy()
                  
                if len(df_C) > 0:
                    left_dfc = df_C[np.where(self.distances[:,0] < self.best_value)[0],...].copy()
                    right_dfc = df_C[np.where(self.distances[:,0] >= self.best_value)[0],...].copy()                
                else:
                    left_dfc = []
                    right_dfc = []
                
                
                left_Y = Y[np.where(self.distances[:,0] < self.best_value)[0]].copy()
                right_Y = Y[np.where(self.distances[:,0] >= self.best_value)[0]].copy()

                # Creating the left and right nodes
                left = CNN_Tree(
                    left_Y, 
                    left_df, 
                    self.features,
                    self.use_features,
                    
                    C_X = left_dfc,
                    C_features = self.C_features,
                    
                    depth=self.depth + 1, 
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split, 
                    node_type='left_node',
                    rule=f"{self.best_feature} < {round(self.best_value, 3)} at time {self.best_time}",
                    max_gain = self.max_gain,

                    ## CNN hyper parameters
                    standardize = self.standardize, ## this is not batch or layer normalization, but instead we normalize the data at each node
                    kernel_size = self.kernel_size, ## Kernel size for the CNN
                    avg_smooth_size = self.avg_smooth_size, ## Smooth the Time Series Signal with moving average (Window_Size)
                    n_ensembles = self.n_ensembles, ## Number of CNN filters to learn
                    drop_pool = self.drop_pool, ## Stochasistiy for the global max during CNN Training (Percentage of time steps randomly masked)
                    epochs = self.epochs, ## Number of epochs
                    learning_rate = self.learning_rate, ## Learning rate for the CNN
                    class_weights = self.class_weights, ## class weights for CNN and logistic regression, can be 'none' or 'balanced'
                    first_derivative = self.first_derivative,
                    ## time hyper parameters
                    time_steps = self.time_steps, ## Time steps, either Proportion Between (0,1) or Integer > 0 and < X.shape[1]
                    only_after = self.only_after, ## Whether to only look after the time step, or before as well
                    weighted_distance = self.weighted_distance, ## IF True average above threshold weighted by proportion of full sequence above threshold
                    distance_function = self.distance_function, ## can be mean or median to calculate distance above threshold 

                    )

                self.left = left 
                self.left.grow_tree()

                right = CNN_Tree(
                    right_Y, 
                    right_df, 
                    self.features,
                    self.use_features,

                    
                    C_X = right_dfc,
                    C_features = self.C_features,                   

                    depth=self.depth + 1, 
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split,
                    node_type='right_node',
                    rule=f"{self.best_feature} >= {round(self.best_value, 3)} at time {self.best_time}",
                    max_gain = self.max_gain,

                    ## CNN hyper parameters
                    standardize = self.standardize, ## this is not batch or layer normalization, but instead we normalize the data at each node
                    kernel_size = self.kernel_size, ## Kernel size for the CNN
                    avg_smooth_size = self.avg_smooth_size, ## Smooth the Time Series Signal with moving average (Window_Size)
                    n_ensembles = self.n_ensembles, ## Number of CNN filters to learn
                    drop_pool = self.drop_pool, ## Stochasistiy for the global max during CNN Training (Percentage of time steps randomly masked)
                    epochs = self.epochs, ## Number of epochs
                    learning_rate = self.learning_rate, ## Learning rate for the CNN
                    class_weights = self.class_weights, ## class weights for CNN and logistic regression, can be 'none' or 'balanced'
                    first_derivative = self.first_derivative,

                    ## time hyper parameters
                    time_steps = self.time_steps, ## Time steps, either Proportion Between (0,1) or Integer > 0 and < X.shape[1]
                    only_after = self.only_after, ## Whether to only look after the time step, or before as well
                    weighted_distance = self.weighted_distance, ## IF True average above threshold weighted by proportion of full sequence above threshold
                    distance_function = self.distance_function, ## can be mean or median to calculate distance above threshold 
                    )

                self.right = right
                self.right.grow_tree()

    def print_info(self, width=1):
        """
        Method to print the infromation about the tree
        """
        # Defining the number of spaces 
        const = int(self.depth * width ** 1.5)
        spaces = "-" * const
        
        if self.node_type == 'root':
            print("Root")
        else:
            print(f"|{spaces} Split rule: {self.rule}")
        print(f"{' ' * const}   | GINI impurity of the node: {round(self.gini_impurity, 2)}")
        print(f"{' ' * const}   | Class distribution in the node: {self.counts}")
        print(f"{' ' * const}   | Predicted class: {self.yhat}")   
        print(f"{' ' * const}   | Predicted class: {self.prob_class_1}")   

        
        
    def print_tree(self):
        """
        Prints the whole tree from the current node to the bottom
        """
        self.print_info() 
        
        if self.left is not None: 
            self.left.print_tree()
        
        if self.right is not None:
            self.right.print_tree()
            
    def return_child(self, tree_frame,id_ = 0,parent_id = None): 
        
        if self.node_type == "root":
            sign = None
        if self.node_type == "left_node":
            sign = '<'
        if self.node_type == "right_node":
            sign = '>='
            
        new_id = id_
        tree_frame.loc[len(tree_frame.index)] = [id_,self.best_feature,self.best_value,parent_id,self.yhat,self.prob_class_1,self.rule,self.node_type,self.counts[0],self.counts[1]]
        if self.left is not None:
            new_id = id_ + 1
            tree_frame,new_id = self.left.return_child(tree_frame,id_=new_id,parent_id = id_)
            
        if self.right is not None:
            new_id = new_id + 1
            tree_frame,new_id = self.right.return_child(tree_frame,id_ = new_id,parent_id = id_)
        
        return tree_frame, new_id
            
    def get_tree_structure(self, width=1):
        """
        Method to print the infromation about the tree
        """
        frame = pd.DataFrame(columns = ["Id","Feature","Value","Parent_Id","Prediction","Probability","Rule","Type","Count_0","Count_1"])
        tree_frame, id_ = self.return_child(frame)
        return tree_frame

    def predict(self, X: np.array, X_C = None, show_inference = False):
        """
        Batch prediction method
        """
        predictions = []
        prediction_dictionary = {}
        probabilities = []
        if X_C is None:
            X_C = np.zeros(X.shape)
        for i in range(X.shape[0]):
            pred, prob, prediction_dictionary = self.predict_obs(X[i:(i + 1),...],X_C[i:(i+1),...],i,prediction_dictionary, show_inference)
            predictions.append(pred)     
            probabilities.append(prob)
        
        predictions = np.array(predictions).reshape(-1,1)
        probabilities = np.array(probabilities).reshape(-1,1)
        predictions = np.concatenate((predictions,probabilities),axis = 1)
        return predictions,prediction_dictionary

    def predict_obs(self, x_obs,xc_obs,id_,prediction_dictionary,show_inference) -> int:
        """
        Method to predict the class given a set of features
        """
        cur_node = self
        sub_dict = {}
        step = 0
        while cur_node.depth < cur_node.max_depth:
            # Traversing the nodes all the way to the bottom
            best_feature = cur_node.best_feature
            best_value = cur_node.best_value
            best_model = cur_node.best_model
            best_time = cur_node.best_time
            best_sign = cur_node.best_before_or_after
            if cur_node.standardize:
                x_obs_norm = (x_obs - cur_node.mean)/(cur_node.std + 1e-9)
            else:
                x_obs_norm = x_obs
            if best_feature is not None:
                if cur_node.is_cat:
                    f = [i for i,v in enumerate(self.C_features) if v == best_feature][0]
                    out = xc_obs[:,f:(f+1)]
                    line = np.zeros((1,1))
                    raw = np.zeros((1,1,1))
                    activations = np.zeros((1,1,1))
                else:
                    f = [i for i,v in enumerate(self.features) if v in best_feature]#[0]
                    to_get = self.X.shape[1] - best_time
                    val_data_loader = create_prediction_data_loaders(x_obs_norm,f)
                    distance,raw,activations = self.get_distance_above_boundary(best_model, val_data_loader, cur_node.best_lr_boundary)
                    line = cur_node.best_lr_boundary


                if cur_node.n < cur_node.min_samples_split:
                    break 
                if (distance < best_value):
                    if self.left is not None:
                        cur_node = cur_node.left
                else:
                    if self.right is not None:
                        cur_node = cur_node.right  
                sub_dict[step] = {
                    "Dist": list(distance.astype('object')[0]),
                    "Raw": list(raw.astype('object')[0,...,0]),
                    "Transform": list(activations.astype('object')[0,...,0]),
                    "feature": best_feature,
                    "time": best_time,
                    "line": line.astype(object)[0],
                    "rule": cur_node.rule,
                    "dir": cur_node.node_type,
                    "y": cur_node.yhat,
                    "prob": cur_node.prob_class_1,
                    "mean": list(cur_node.mean.astype('object')),
                    "std": list(cur_node.std.astype('object'))
                }     
                step = step + 1                                                   
            else:
                break
#         print("Predicted " + str(cur_node.yhat))
#         print("next prediction")
        prediction_dictionary[id_] = sub_dict
        return cur_node.yhat,cur_node.prob_class_1, prediction_dictionary