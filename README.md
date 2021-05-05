# log-analyzer

## Introduction

This repository is the main project page for research into log messages and their potential for determining system state information. The following directories, listed and defined in no particular order, comprise the project: 

jupyter: used to build the docker container. (section 1)
notebook: This directory will contains the necessary code to standup the jupyter notebook. (section 2)
data: All databases and other source data will be stored here, note this is not where models and other serialized derived data is to be stored.
reults: Data which is derivative of information from the data directory will be kept here. (section 3)
doc: Any useful documentation will be kept in this directory. This includes research papers and notes. 
preprocessing: The code for the pre-processing pipeline. (section 4)
training: The transformer code is maintained here. Note this is not where saved models are stored, check the results folder. (section 5)

## 0. First Steps 

Hey there! In this section I will describe the steps to getting the jupyter container running. If you are experienced and just want the tldr then here it is: make sure there are database files in the data dir (message me if you need these files) and then stand up the jupyter container using docker-compose (use the --build flag: sudo docker-compose up --build jupyter). Below is the gentle introduction.

I highly recommend setting up ssh for GitHub. The use of passwords will soon be depreciated. Here are some links on how to do this. Note you will need to setup an ssh key for each computer you wish to use and the key is repository agnostic (meaning you will not need a new key for each repository you access). 

Some information on SSH: https://www.hostinger.com/tutorials/ssh-tutorial-how-does-ssh-work
GitHub's guide to setting up an ssh key for your machine: https://docs.github.com/en/enterprise-server@3.0/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account

I also highly recommend using some flavor of Linux. Docker is heavily used in this project and runs better on Linux since it does not require a VM unlike macOS and Windows 10. Although this is highly controversial I would recommend using Ubuntu 21.04 or 20.04 LTS. If Cannonical disgust you then I would recommend using MX Linux or Manjoro. 

All instructions in this document will assume the use of linux, if you are using WSL2 or macOS there may be slight alterations needed to make the code work. I will write about this at the end of the section. Also at the end will be a quick introduction on setting up Docker on WSL2 (Windows) and macOS. If you need further help please message me and I will try to help where I can. 

Please note that if you intend to use CUDA to leverage your GPU then linux is a requirement. I know that GPU passthrough is a thing on the Window's developer ring and thus technically useable with WSL2 using Nvidia's beta drivers. I would advise against this route unless you truly know what you are doing and are willing to accept the risk of an unstable Windows 10. 

From here on I will assume you are using Ubuntu and thus aptitude. If you are using yum or pacman just substitute those commands or using Docker's documentation for installing on those distributions. 

To start you will need docker. Check to see if any previous version of Docker is lingering around - we will need to remove it if so. 

```bash 
sudo apt-get remove docker docker-engine docker.io containerd runc
```

## 1. jupyter 

This is the main directory for experimentation. It houses all the code necessary to build the jupyter docker container. The notebooks, results, data, and doc directories are mapped to this container through docker-compose for serializing/deserializing objects, loading trained models, and modifying LaTeX reports. 

## 2. notebooks 

This is source folder for all Jupyter notebooks. As Jupyter is the primary tool for experimentation this folder houses all experimental code. Updates and modifications to the pre-processing and training folders more than likely originate from this directory. Currently there is only the longruntransformer.ipynb notebook in this directory which holds the experimental transformer code. I will soon add a playground.ipynb notebook for throwaway code. 

## 3. results 

All trained models, pickeled intermediary results, and tensorflow checkpoints, and static graphs will be kept here. This directory is mapped to the jupyter container through docker-compose for loading and saving objects. 

## 4. preprocessing 

## 5. training