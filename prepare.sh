#!/usr/bin/env bash
sudo apt-get install git-lfs --yes
git config --global credential.helper store

python3 -m venv app-env
source app-env/bin/activate
echo "source $PWD/app-env/bin/activate" >> ~/.bashrc

pip install -r requirements.txt