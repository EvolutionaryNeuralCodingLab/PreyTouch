#!/bin/bash

set -e


if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
fi

sudo sh -c "truncate -s 0 /var/lib/docker/containers/$(docker-compose ps -q $1)/*-json.log"

#clear () {
#        local LOGPATH=$(docker inspect --format='{{.LogPath}}' $(docker-compose ps -q $1))
#        docker run -it --rm --privileged --pid=host alpine:latest nsenter -t 1 -m -u -n -i -- truncate -s0 $LOGPATH
#}
#
#clear
