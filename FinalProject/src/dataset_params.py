import numpy as np
import copy
import os 
import datetime
import time
import scipy.io as sio
from custom_funcs import generate_data
import pandas as pd
from sklearn.utils import resample
def Dataset_params(dataset="COIL20"):
    if dataset == "synthetic":
        data_path = "data/syndata200x50.csv"
        data = np.genfromtxt(data_path, dtype='f4', delimiter=',')
        full_data = copy.deepcopy(data)
        input_shape = data.shape
        batch_size = input_shape[0]
        flat_layer_size = [input_shape[0]]
        enc_layer_size = [40]
        deco_layer_size = [input_shape[-1]]
        total_datapoints = input_shape[0]*input_shape[1]
        K = num_class = 4
        reg1 = 1  # recon loss
        reg2 = 0.001  #auto-encoder loss
        alpha1, alpha2 = 1,1
        d = 3
        lr = 0.005
        labels = np.repeat(np.arange(1, 5), 50)
        true_labels = labels.flatten()
        rank = num_class * d
        return data,full_data,input_shape,batch_size,total_datapoints,flat_layer_size,enc_layer_size,\
            deco_layer_size, None, None, K, reg1, reg2, alpha1, alpha2, d, lr, rank, true_labels

    elif dataset == "COIL20":
        data = sio.loadmat("data/COIL20.mat")
        Img = data['fea']
        Label = data['gnd']
        Img = np.reshape(Img,(Img.shape[0],32,32,1))
        batch_size = 72*20


        K = num_class = 20 #how many class we sample
        num_sa = 72

        alpha1 = 1
        alpha2 = 8
        d = 12
        reg1 = 1.0
        reg2 = 1e-5
        rank = num_class * d
        lr = 4e-2
        coil20_all_subjs = Img
        data = coil20_all_subjs.astype(float)	
        label_all_subjs = Label
        label_all_subjs = label_all_subjs - label_all_subjs.min() + 1    
        true_labels = np.squeeze(label_all_subjs)  	
        full_data = copy.deepcopy(coil20_all_subjs) # Deep copy of full data
        # flat_array = np.reshape(full_data,[1440,1024])
        # np.savetxt("full_data.csv", flat_array, delimiter=",")
        input_shape = np.shape(coil20_all_subjs)
        total_datapoints = input_shape[0]*input_shape[1]*input_shape[2]
        kernel_size = [3]
        flat_layer_size = [batch_size ] # Flat Layer Neurons Size
        enc_layer_size = [15] # Encoder Layer channels
        deco_layer_size = [input_shape[-1]] # Decoder Layer channels
        output_padding = [1]
        return data,full_data,input_shape,batch_size,total_datapoints,flat_layer_size,enc_layer_size,\
            deco_layer_size,kernel_size,output_padding, K, reg1, reg2, alpha1, alpha2, d, lr, rank, true_labels


    elif dataset=="EYaleB":
        n_input = [48,42]
        data = sio.loadmat("data/YaleBCrop025.mat")
        img = data['Y']
        I = []
        Label = []
        for i in range(img.shape[2]):
            for j in range(img.shape[1]):
                temp = np.reshape(img[:,j,i],[42,48])
                Label.append(i)
                I.append(temp)

        I = np.array(I)
        Label = np.array(Label[:])
        Img = np.transpose(I,[0,2,1])
        Img = np.expand_dims(Img[:],3)

        # all_subjects = [10, 15, 20, 25, 30, 35, 38]
        num_class = 20 #how many class we sample
        num_sa = 64

        batch_size = num_class * num_sa
        alpha1 = 1 #thrC
        alpha2 = 3.5 #post_procC
        d = 10
        reg1 = 1.0
        reg2 = 1e-5
        rank = num_class * d
        lr = 5e-2
        i = 0
        face_10_subjs = np.array(Img[64*i:64*(i+num_class),:])
        data = face_10_subjs.astype(float)        
        label_10_subjs = np.array(Label[64*i:64*(i+num_class)]) 
        label_10_subjs = label_10_subjs - label_10_subjs.min() + 1
        true_labels = np.squeeze(label_10_subjs)    
        K = true_labels.max()        
        full_data = copy.deepcopy(data) # Deep copy of full data

        input_shape = np.shape(data)
        total_datapoints = input_shape[0]*input_shape[1]*input_shape[2]

        
        kernel_size = [5,3,3]

        flat_layer_size = [batch_size ] # Flat Layer Neurons Size
        enc_layer_size = [10, 20, 30] # Encoder Layer channels
        deco_layer_size = [20, 10, input_shape[-1]] # Decoder Layer channels
        output_padding = [1,(0,1), 1]
        return data,full_data,input_shape,batch_size,total_datapoints,flat_layer_size,enc_layer_size,\
            deco_layer_size,kernel_size,output_padding , K, reg1, reg2, alpha1, alpha2, d, lr, rank, true_labels

    elif dataset=="ORL":
        data = sio.loadmat("data/ORL_32x32.mat")
        Img = data['fea']
        Label = data['gnd']
        n_input = [32, 32]
        Img = np.reshape(Img,[Img.shape[0],n_input[0],n_input[1],1]) 
        
        num_class = 40
        num_sa = 10
        i = 0
        face_10_subjs = np.array(Img[10*i:10*(i+num_class),:])
        data = face_10_subjs.astype(float)        
        label_10_subjs = np.array(Label[10*i:10*(i+num_class)]) 
        label_10_subjs = label_10_subjs - label_10_subjs.min() + 1
        true_labels = np.squeeze(label_10_subjs) 
        batch_size = num_class * num_sa
        K = true_labels.max()
        alpha1 = 0.2 #thrC
        alpha2 = 3.5 #post_procC
        d = 10
        reg1 = 1.0
        reg2 = 1e-5
        rank = num_class * 1
        lr = 4e-2


        full_data = copy.deepcopy(data) # Deep copy of full data

        input_shape = np.shape(data)
        total_datapoints = input_shape[0]*input_shape[1]*input_shape[2]

        kernel_size = [3, 3, 3]

        flat_layer_size = [batch_size ] # Flat Layer Neurons Size
        enc_layer_size = [3, 3, 5] # Encoder Layer channels
        deco_layer_size = [3, 3, input_shape[-1]] # Decoder Layer channels
        output_padding = [1, 1, 1]
        return data,full_data,input_shape,batch_size,total_datapoints,flat_layer_size,enc_layer_size,\
            deco_layer_size, kernel_size, output_padding, K, reg1, reg2, alpha1, alpha2, d, lr, rank, true_labels

    elif dataset == "BostonHousing":
        data_path = "data/BostonHousing2.csv"
        data = pd.read_csv(data_path, skiprows=1).to_numpy()
        full_data = copy.deepcopy(data)
        input_shape = data.shape
        batch_size = input_shape[0]
        flat_layer_size = [input_shape[0]]
        enc_layer_size = [12, 10]
        deco_layer_size = [12, input_shape[-1]]
        total_datapoints = input_shape[0]*input_shape[1]
        K = num_class = 1
        reg1 = 2  # recon loss
        reg2 = 1e-5  #auto-encoder loss
        alpha1, alpha2 = 1,1
        d = 3
        lr = 5e-2
        true_labels = None
        rank = num_class * d
        return data,full_data,input_shape,batch_size,total_datapoints,flat_layer_size,enc_layer_size,\
            deco_layer_size, None, None, K, reg1, reg2, alpha1, alpha2, d, lr, rank, true_labels
    
    elif dataset == "PeriodChanger":
        data_path = "data/period-changer.csv"
        data_in = pd.read_csv(data_path).to_numpy()
        data = data_in[:,:-1]
        labels = data_in[:,-1]

        # Label processing to ensure they start from 1
        label_all_subjs = labels
        label_all_subjs = label_all_subjs - label_all_subjs.min() + 1    
        true_labels = np.squeeze(label_all_subjs) 

        full_data = copy.deepcopy(data)
        input_shape = data.shape
        batch_size = input_shape[0]
        flat_layer_size = [input_shape[0]]
        enc_layer_size = [1024,]
        deco_layer_size = [input_shape[-1]]
        total_datapoints = input_shape[0]*input_shape[1]
        K = num_class = 2
        reg1 = 1  # recon loss
        reg2 = 1e-5  #auto-encoder loss
        alpha1, alpha2 = 1,1
        d = 3
        lr = 7e-3
        true_labels = labels
        rank = 6
        return data,full_data,input_shape,batch_size,total_datapoints,flat_layer_size,enc_layer_size,\
            deco_layer_size, None, None, K, reg1, reg2, alpha1, alpha2, d, lr, rank, true_labels

    elif dataset == "DSDD":
        data_in = sio.loadmat("data/Dataset_for_Sensorless_Drive_diagnosis.mat")
        data = data_in['fea']
        labels = data_in['lab']
        label_all_subjs = labels.T
        label_all_subjs = label_all_subjs - label_all_subjs.min() + 1    
        true_labels = np.squeeze(label_all_subjs) 
        full_data = copy.deepcopy(data)
        input_shape = data.shape
        batch_size = input_shape[0]
        flat_layer_size = [input_shape[0]]
        enc_layer_size = [40]
        deco_layer_size = [input_shape[-1]]
        total_datapoints = input_shape[0]*input_shape[1]
        K = num_class = 11
        reg1 = 1  # recon loss
        reg2 = 1e-5  #auto-encoder loss
        alpha1, alpha2 = 1,1
        d = 4 #need to figure out
        lr = 7e-3 #need to figure out
        true_labels = labels
        rank = K * d
        return data,full_data,input_shape,batch_size,total_datapoints,flat_layer_size,enc_layer_size,\
            deco_layer_size, None, None, K, reg1, reg2, alpha1, alpha2, d, lr, rank, true_labels
    
    elif dataset == "HeartDisease":

        data_path = "data/heartdisease.csv"
        data_in = pd.read_csv(data_path).to_numpy()
        data = data_in[:,:-1]
        labels = data_in[:,-1]
        # Label processing to ensure they start from 1
        label_all_subjs = labels
        label_all_subjs = label_all_subjs - label_all_subjs.min() + 1    
        true_labels = np.squeeze(label_all_subjs) 

        # Data for further processing
        full_data = copy.deepcopy(data)
        input_shape = data.shape
        batch_size = input_shape[0]
        flat_layer_size = [input_shape[0]]
        enc_layer_size = [12]
        deco_layer_size = [input_shape[-1]]
        total_datapoints = input_shape[0]*input_shape[1]
        K = num_class = 5
        reg1 = 1  # recon loss
        reg2 = 1e-5  # auto-encoder loss
        alpha1, alpha2 = 1, 1
        d = 1
        lr = 7e-3
        rank = 6

        return data, full_data, input_shape, batch_size, total_datapoints, flat_layer_size, enc_layer_size, \
            deco_layer_size, None, None, K, reg1, reg2, alpha1, alpha2, d, lr, rank, true_labels
    
    
    elif dataset == "HARUS":
        data_in = sio.loadmat("data/HARUS.mat")
        data = data_in['fea']
        labels = data_in['lab']
        label_all_subjs = labels
        label_all_subjs = label_all_subjs - label_all_subjs.min() + 1    
        true_labels = np.squeeze(label_all_subjs) 
        full_data = copy.deepcopy(data)
        input_shape = data.shape
        batch_size = input_shape[0]
        flat_layer_size = [input_shape[0]]
        enc_layer_size = [512, 256]
        deco_layer_size = [512, input_shape[-1]]
        total_datapoints = input_shape[0]*input_shape[1]
        K = num_class = 6
        reg1 = 1  # recon loss
        reg2 = 1e-5  #auto-encoder loss
        alpha1, alpha2 = 1,1
        d = 20
        lr = 7e-3 #need to figure out
        true_labels = labels
        rank = K * d
        return data,full_data,input_shape,batch_size,total_datapoints,flat_layer_size,enc_layer_size,\
            deco_layer_size, None, None, K, reg1, reg2, alpha1, alpha2, d, lr, rank, true_labels
    
    elif dataset == "Flowers":
        data_in = sio.loadmat(r"data/flowers.mat")
        Img = data_in['fea'] # 8189 x 32 x 32 x 3
        Label = data_in['lab']
        label_all_subjs = Label
        label_all_subjs = label_all_subjs - label_all_subjs.min() + 1    
        true_labels = np.squeeze(label_all_subjs) 
        input_shape = np.shape(Img)
        batch_size = input_shape[0]
        data = Img.astype(float)	
        label_all_subjs = Label
        label_all_subjs = label_all_subjs - label_all_subjs.min() + 1    
        true_labels = np.squeeze(label_all_subjs)  	
        full_data = copy.deepcopy(data) # Deep copy of full data

        K = num_class = 102
        alpha1 = 1
        alpha2 = 8
        d = 2 #2 or 1
        reg1 = 1.0
        reg2 = 1e-5
        rank = num_class * d 
        lr = 1e-3 #need to figure out

        # flat_array = np.reshape(full_data,[1440,1024])
        # np.savetxt("full_data.csv", flat_array, delimiter=",")
        total_datapoints = input_shape[0]*input_shape[1]*input_shape[2]*3
        kernel_size = [3, 3, 3]
        flat_layer_size = [batch_size ] # Flat Layer Neurons Size
        enc_layer_size = [32, 64, 128] # Encoder Layer channels
        deco_layer_size = [64, 32, input_shape[-1]] # Decoder Layer channels
        output_padding = [1, 1, 1]
        return data,full_data,input_shape,batch_size,total_datapoints,flat_layer_size,enc_layer_size,\
            deco_layer_size,kernel_size,output_padding, K, reg1, reg2, alpha1, alpha2, d, lr, rank, true_labels
    
    elif dataset == "Oxford_Pet":
        data_in = sio.loadmat(r"data/oxford_pet.mat")
        Img = data_in['fea'] # 7349 x 32 x 32 x 3
        Label = data_in['lab']
        label_all_subjs = Label
        label_all_subjs = label_all_subjs - label_all_subjs.min() + 1    
        true_labels = np.squeeze(label_all_subjs) 
        input_shape = np.shape(Img)
        batch_size = input_shape[0]
        data = Img.astype(float)	
        label_all_subjs = Label
        label_all_subjs = label_all_subjs - label_all_subjs.min() + 1    
        true_labels = np.squeeze(label_all_subjs)  	
        full_data = copy.deepcopy(data) # Deep copy of full data

        K = num_class = 37
        alpha1 = 1
        alpha2 = 8
        d = 3 
        reg1 = 1.0
        reg2 = 1e-5
        rank = num_class * d 
        lr = 9e-3 #need to figure out

        # flat_array = np.reshape(full_data,[1440,1024])
        # np.savetxt("full_data.csv", flat_array, delimiter=",")
        total_datapoints = input_shape[0]*input_shape[1]*input_shape[2]*3
        kernel_size = [3]
        flat_layer_size = [batch_size ] # Flat Layer Neurons Size
        enc_layer_size = [15] # Encoder Layer channels
        deco_layer_size = [input_shape[-1]] # Decoder Layer channels
        output_padding = [1]
        return data,full_data,input_shape,batch_size,total_datapoints,flat_layer_size,enc_layer_size,\
            deco_layer_size,kernel_size,output_padding, K, reg1, reg2, alpha1, alpha2, d, lr, rank, true_labels
    
    elif dataset == "Novel_dataset":
        # Refer ORL, COIL20, EYALE for convolutional layer settings
        # Refer Flowers, Oxford_Pet for RGB dataset settings
        # Refer Heart Disease for only completion settings ( IF data is unlabeled)
        # Refer Boston Housing for simplest modelling
        # Refer the preset dataset settings based on dataset type (.mat or .csv)

        # ─── Load .mat and extract features & labels ─────────────────────
        data_in = sio.loadmat("noveldata.mat")
        Img = data_in['fea']         
        Label = data_in['lab'].flatten()
        # normalize labels to 1…N
        true_labels = Label - Label.min() + 1 # Set None if unlabeled

        # ─── Meta‐info ────────────────────────────────────────────────────
        full_data = data_in['fea'].astype(float).copy()
        input_shape = Img.shape      # (n_samples, height, width, chan)
        batch_size  = input_shape[0]
        total_datapoints = Img.size

        # ─── Clustering & regularization settings ────────────────────────
        num_class = 37               # number of distinct classes (set to 0 if unlabeled)
        K = num_class
        alpha1 = 1.0                 # thrC
        alpha2 = 1.0                 # post_procC
        d = 3                        # subspace dimension

        # ─── Optimization & model hyperparams ────────────────────────────
        reg1 = 1.0                   # reconstruction loss
        reg2 = 1e-5                  # auto-encoder loss ( only for SSC)
        rank = num_class * d
        lr = 2e-5 # sensitive to learning rate so try different values
        # ─── Autoencoder architecture sizes ──────────────────────────────
        kernel_size      = None              # kernel size (e.g. [3, 3, 3] for 3D conv)
        flat_layer_size  = [batch_size]      # size of the flattened bottleneck
        enc_layer_size   = [15]              # encoder channels size
        deco_layer_size  = [input_shape[-1]] # decoder channels size
        output_padding   = None              # output padding (e.g. [1, 1, 1] for 3D conv)              

        return data,full_data,input_shape,batch_size,total_datapoints,flat_layer_size,enc_layer_size,\
            deco_layer_size,kernel_size,output_padding, K, reg1, reg2, alpha1, alpha2, d, lr, rank, true_labels


def create_log(dataset):
    timestamp = str(datetime.datetime.fromtimestamp(int(time.time())))
    formatted_timestamp = timestamp.replace(" ", "__").replace(":", "-")
    log_folder = "exp_" + dataset
    logs_path = os.path.join("./logs/", log_folder)
    return logs_path

