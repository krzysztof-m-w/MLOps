#!/bin/bash

sudo apt update
sudo apt install -y docker.io
sudo systemctl enable --now docker
usermod -aG docker $USERNAME
