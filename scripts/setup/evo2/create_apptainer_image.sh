#!/bin/bash
set -e

# clone evo2 repository.
git clone https://github.com/ArcInstitute/evo2.git

# substitute custom dockerfile.
mv evo2/Dockerfile evo2/Dockerfile.ORIGINAL
cp ./Dockerfile evo2/

# build the image.
cd evo2
docker build -t evo2_gem:latest .
docker save evo2_gem:latest -o evo2_gem.tar
apptainer build evo2_gem.sif docker-archive://evo2_gem.tar


echo "Successfully built Apptainer image: evo2_gem.sif"
