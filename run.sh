#!/bin/bash
source bin/activate
cd ./modules
python main.py -t=True
./modules/extractTrainLog.sh

python main.py -t=False
./extractTestLog.sh
