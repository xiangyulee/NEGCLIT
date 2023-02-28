# Developer Guide
## Repo Structure Overview
    ├──agents                       #Files for different online continual learning algorithms
        ├──base.py                      #Abstract class for algorithms
        ├──agem.py                      #File for A-GEM
        ├──cndpm.py                     #File for CN-DPM
        ├──ewc_pp.py                    #File for EWC++
        ├──exp_replay.py                #File for ER, MIR and GSS
        ├──gdumb.py                     #File for GDumb
        ├──iCaRL.py                     #File for iCaRL
        ├──lwf.py                       #File for LwF
        ├──scr.py                       #File for SCR

    ├──continuum                    #Files for create the data stream objects
        ├──dataset_scripts              #Files for processing each specific dataset
            ├──dataset_base.py              #Abstract class for dataset
            ├──cifar10.py                   #File for CIFAR10
            ├──cifar100,py                  #File for CIFAR100
            ├──core50.py                    #File for CORe50
            ├──mini_imagenet.py             #File for Mini_ImageNet
            ├──openloris.py                 #File for OpenLORIS
        ├──continuum.py             
        ├──data_utils.py
        ├──non_stationary.py

    ├──models                       #Files for backbone models
        ├──ndpm                         #Files for models of CN-DPM 
            ├──...
        ├──pretrained.py                #Files for pre-trained models
        ├──resnet.py                    #Files for ResNet

    ├──utils                        #Files for utilities
        ├──buffer                       #Files related to buffer
            ├──aser_retrieve.py             #File for ASER retrieval
            ├──aser_update.py               #File for ASER update
            ├──aser_utils.py                #File for utilities for ASER
            ├──buffer.py                    #Abstract class for buffer
            ├──buffer_utils.py              #General utilities for all the buffer files
            ├──gss_greedy_update.py         #File for GSS update
            ├──mir_retrieve.py              #File for MIR retrieval
            ├──random_retrieve.py           #File for random retrieval
            ├──reservoir_update.py          #File for random update

        ├──global_vars.py               #Global variables for CN-DPM
        ├──io.py                        #Code related to load and store csv or yarml
        ├──kd_manager.py                #File for knowledge distillation
        ├──name_match.py                #Match name strings to objects 
        ├──setup_elements.py            #Set up and initialize basic elements
        ├──utils.py                     #File for general utilities

    ├──config                       #Config files for hyper-parameters tuning
        ├──agent                        #Config files related to agents
        ├──data                         #Config files related to dataset

        ├──general_*.yml                #General yml (fixed variables, not tuned)
        ├──global.yml                   #paths to store results 