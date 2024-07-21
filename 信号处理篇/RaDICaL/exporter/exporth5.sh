#!/bin/bash

DATAPATH=/mnt/datasets-drive
CODEPATH=$(pwd)/../

docker build -f dockerfiles/Dockerfile ./ --tag ros-radar


function run_export {
    echo "$1" + "$2" >> h5export.log
    docker run \
        -e TMPDIR=$TMPDIR \
        --mount type=bind,source=/mnt,target=/mnt \
        --mount type=bind,source="$CODEPATH",target=/code  \
        -w /code/exporter \
        -u $(id -u):$(id -g) \
        ros-radar python rosbag_to_h5.py \
        --bagfile "$1" \
        --h5file "$1".export.h5 \
        --radarcfg "$2" \
}

function get_radarcfg {
    if [[ $1 == *"indoor"* ]]; then
        echo "$DATAPATH/radarcfgs/indoor_human_rcs.cfg"
    elif [[ $1 == *"30m"* ]]; then
        echo "$DATAPATH/radarcfgs/outdoor_human_rcs_30m.cfg"
    elif [[ $1 == *"50m"* ]]; then
        echo "$DATAPATH/radarcfgs/outdoor_human_rcs_50m.cfg"
    else
        return -1
    fi
}


find $DATAPATH -name "*.bag" | while read -r file
do
    #run_generate "$file"
    if get_radarcfg $file; then
        echo $file "<--" $(get_radarcfg "$file")
        run_export $file $(get_radarcfg "$file")
    else
        echo Skipping $file
    fi
done
