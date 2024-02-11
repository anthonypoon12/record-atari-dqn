#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import ipdb
import numpy as np
import os,glob
import csv
from PIL import Image
import pandas as pd
from collections import defaultdict

# Super Logger is a class to manage ActivationLoggers
class SuperLogger():
    # Holds a list of loggers, and uses the outputs to create a list to place into a csv
    def __init__(self, csvFilePath):
        self.loggers = []
        self.csvFilePath = csvFilePath
    
    # addLogger returns the logger, allowing us to directly register the hook 
    def addLogger(self):
        logger = ActivationLogger(self)
        self.loggers.append(logger)
        return logger
    
    def checkAddToCSV(self):
        # If all loggers have an output, we can place into the csv
        # After placing into csv, we can reset all outputs
        newRow = []
        for logger in self.loggers:
            if logger.output == None:
                return
            else:
                newRow.append(logger.output)
        with open(self.csvFilePath, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(newRow)
            self.setAllLoggers(None)

    def setAllLoggers(self, value):
        for logger in self.loggers:
            logger.output = value
        
# Activation Logger allows us to hold onto the output value
class ActivationLogger:
    def __init__(self, superLogger):
        self.superLogger = superLogger
        self.output = None

    def __call__(self, module, input, output):
            self.output = output.tolist()
            self.superLogger.checkAddToCSV()

class Record():
    def __init__(self,seed,output_dir):
        self.seed=seed
        
        output_dir = output_dir+'/model_data/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        # Delete previous data in output_dir
        previous_files = glob.glob(output_dir+'**/*.npy')
        for file in previous_files:
            os.remove(file)
            
        self.output_dir=output_dir

        self.imageCounter = 0

        self.activationHookFiles = []

        self.superLogger = None
        
    def grab_w_n_b(self,agent,episode):
        """ Saves weight and bias information for each agent as a npy file """
        for name, param in agent.policy_network.named_parameters():
            #ipdb.set_trace()
            if 'fc' in name:
                output_name=self.output_dir+'/'+name.replace('.','_')+'/'
                if not os.path.exists(output_name):
                    os.mkdir(output_name)
                output_name=self.output_dir+'/'+name.replace('.','_')+'/'+str(self.seed)+'_'+str(episode)+"_"+name.replace('.','_')+'.npy'
                np.save(output_name,param.cpu().detach().numpy())
    
    def custom_sort(self,filename):
        try:
            _,_,_,_,number,_,_,_=filename.split('_')
        except:
            ipdb.set_trace()
        return int(number)
    
    def concat_w_n_b(self):
        """ Concatenates each episode's weights and biases for current agent """
        subfolders=glob.glob(self.output_dir+'/*')
        for folder in subfolders:
            for i,file in enumerate(sorted(glob.glob(folder+'/*.npy'),key=self.custom_sort)):
                if i==0:
                    fmat=np.load(file)
                    fmat=fmat[...,np.newaxis]
                    os.remove(file)
                else:
                    mat_oh=np.load(file)
                    mat_oh=mat_oh[...,np.newaxis]
                    os.remove(file)
                    fmat=np.concatenate((fmat, mat_oh),axis=-1) 
            output_name=folder+'/concat.npy'
            np.save(output_name,fmat)

    def add_activation_hook(self, agent):
        # module is essentially the layer

        # Keeps track of layer name and description to add to top of csv
        layerList = []
        descriptionsList = []
        outputPath=self.output_dir+'/'+'activations'+'.csv'
        self.superLogger = SuperLogger(outputPath)
        
        # superLogger will add loggers in correct order
        for name, module in agent.named_modules():
            name = name if name != "" else "no_name_network"
            module.register_forward_hook(self.superLogger.addLogger())
            layerList.append(name)
            descriptionsList.append(str(module))
        
        # Adding first row with layer names
        # I chose w to reset the file
        with open(outputPath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(layerList)
            writer.writerow(descriptionsList)
            
    def recordObservation(self, observation, episode):
        path = f'{self.output_dir}/Episode{episode}'
        if not os.path.exists(path):
            os.mkdir(path)
        Image.fromarray(observation).save(f'{path}/Image{self.imageCounter}.png')
    
    def buildDF(self):
        # Organizes each folder by its layer name
        # This assumes that each folder is named by "layer_bias" or "layer_weight"
        # We create a dictionary where key is the layer
        # Each value is an array with two folder names, the first being the bias and second being weights
        folderDict = defaultdict(list)
        subfolders=glob.glob(self.output_dir+'/*')

        for folder in subfolders:
            split_string = folder.split('_')
            if split_string[-1] in ['bias', 'weight']:
                folderDict[tuple(folder.split('_')[:-1])].append(folder)
        
        df = pd.DataFrame()

        for key, folders in folderDict.items():
            biasFolder, weightFolder = folders
            # Makes sure that the variables are correct
            if 'weight' not in weightFolder:
                weightFolder, biasFolder = biasFolder, weightFolder

            data = np.load(weightFolder + '/concat.npy')
            bias_data = np.load(biasFolder + '/concat.npy')

            datashape = data.shape

            next_layer_idx, neuron_idx, episode_idx = np.indices(datashape)

            # reshaped_data = data.reshape(-1, datashape[2])
            reshaped_bias_data = bias_data.reshape(-1)

            dictionary = {
                "Episode": episode_idx.flatten() + 1,
                "Neuron id": neuron_idx.flatten() + 1,
                "Next Layer Neuron": next_layer_idx.flatten() + 1,
                "Layer": os.path.basename(''.join(key)),
                # "Weight Value": reshaped_data.flatten(),
                "Weight Value": data.flatten(),
                "Bias Value": np.repeat(reshaped_bias_data, datashape[1])
            }

            df2 = pd.DataFrame(dictionary)
            df = pd.concat([df, df2])
        return df
    
# To do --> 
    # Need weights at end of every episode for every neuron along with biases
    # Need recording of activity for each neuron to determien whether neuron is activated by threat
    #   Need Image at same time to determine when "threat" is present.