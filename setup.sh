#!/bin/sh

#https://phoenixnap.com/kb/how-to-install-python-3-ubuntu

#sudo apt-get install python3-venv

# sudo apt-get install nginx

#commented out below scripts temp
mkdir /storage
mkdir /downloads

chmod 777 -R /storage
chmod 777 -R /downloads

#yarn build

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

#python3 -m venv env
##venv\Scripts\activate
#source env/bin/activate

pip3 install -r requirements.txt

#cp ./adapttext.service /etc/systemd/system/adapttext.service

mkdir ./adapttext/optimizer
git clone https://gitlab.com/yathindra/fastai1.git
git clone https://github.com/lessw2020/Best-Deep-Learning-Optimizers.git
cp ./Best-Deep-Learning-Optimizers/diffgrad/diffgrad.py ./adapttext/optimizer/DiffGradOptimizer.py
rm -rf ./Best-Deep-Learning-Optimizers

#cd ..