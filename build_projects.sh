#!/bin/bash

# A script to poetry build and docker build the following list of projects

WORKSPACE="pavai"
#PROJECTS=("talkie_app" "vocei_app" "finetune_app" "finetune_worker1")
PROJECTS=("finetune_app" "finetune_worker1" "finetune_worker2")

# Get all subdirectories
cd projects
for project in ${PROJECTS[@]}; do
    cd $project
    poetry build-project
    docker build -t $WORKSPACE/$project .
    cd ..
done

echo "Done."
