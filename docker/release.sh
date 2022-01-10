#!/bin/bash
set -ex

USERNAME=ebadrian
IMAGE=ss-metadl
VERSION=`cat VERSION`

docker push $USERNAME/$IMAGE:gpu-$VERSION
docker push $USERNAME/$IMAGE:gpu-latest

