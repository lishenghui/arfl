#!/usr/bin/env bash

apt update
echo "set nu" >> ~/.vimrc
apt install -y vim htop wget zip
pip install -r requirements.txt

pushd ./data/shakespeare/ || exit
rm -rf ./data ./meta
#./preprocess.sh -s niid --sf 0.1 -k 0 -t sample -tf 0.8 --smplseed 0


pushd ./preprocess || exit
python download_data.py
popd || exit

popd || exit

pushd ./data/femnist/preprocess || exit
rm -rf ../data ../meta
wget https://www.dropbox.com/s/nhrl2ep4r12nu18/femnist.zip
mkdir -p ../data
unzip -o femnist.zip -d ../data
rm femnist.zip
popd || exit

pushd ./data/cifar10/preprocess || exit
#python get_cifar10.py
popd || exit