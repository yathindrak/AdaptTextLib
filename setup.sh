#!/bin/sh

#https://phoenixnap.com/kb/how-to-install-python-3-ubuntu

#sudo apt-get install python3-venv

#python3 -m venv venv
#venv\Scripts\activate
#source venv/bin/activate

# sudo apt-get install nginx

#commented out below scripts temp
mkdir /storage
mkdir /downloads

# setup instance
#commented out below scripts temp
apt update
#apt install software-properties-common
#add-apt-repository ppa:deadsnakes/ppa -y
#apt update
#apt install python3
#apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
#apt install python3-pip
#apt-get install python3-venv
#install --upgrade pip
#apt-get install git
#
#sudo apt update

# setup libs for the app
#cd api

#python3 -m venv venv
#source venv/bin/activate

pip3 install -r requirements.txt

mkdir ./loretex/optimizer
git clone https://gitlab.com/yathindra/fastai1.git
git clone https://github.com/lessw2020/Best-Deep-Learning-Optimizers.git
cp ./Best-Deep-Learning-Optimizers/diffgrad/diffgrad.py ./loretex/optimizer/DiffGradOptimizer.py
rm -rf ./Best-Deep-Learning-Optimizers

#cd ..