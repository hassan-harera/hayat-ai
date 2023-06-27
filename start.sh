#!/bin/bash
#Docker down
#eval $(minikube docker-env)
docker-compose -f docker-compose.yml down

#Docker up
docker-compose -f ./docker-compose.yml up --build -d
