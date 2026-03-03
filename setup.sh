#!/usr/bin/env bash

set -e

cp .env.example .env

python3.11 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt