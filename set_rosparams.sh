#!/bin/bash

declare -A rosparams

# most-used
rosparams[pedicel_calc_mode]=4 # mode for quaternion calculation (0-4 as of now)
rosparams[which_pedicel]=0 # for selecting which pedicel to target. 0 means uppermost, 1 means the one below. Note that this has to be <= n_pedicels-1

# less-used
rosparams[pedicel_cut_prop]="[0.25,0.50,0.75]" # where the cut point is along the pedicel curve, 0.5 means half way point. 0.8 means somewhere close to the tomato fruit.
rosparams[bbox_top]="1" # the amount of top part of bbox used for determining if there is a pedicelattached to it.
rosparams[ripeness_threshold]="0.1" # less than this value means ripe, else unripe. 
rosparams[ripeness_percentile]="0.25" # the percentile used for removing outliers in ripeness values
rosparams[deg]=2 # degree of polynomial curve fitting
rosparams[max_dist]=800 # max distance from camera center to pedicel in mm (to avoid making attempts for detections too far)
rosparams[curve_length_min]=20 # minimum pedicel length in mm
rosparams[gap_min]=5 # minimum gap required between pedicel and tomato in mm

# instance segmentation thresholds
rosparams[threshold_stem]="0.3"
rosparams[threshold_tomato]="0.1"
rosparams[threshold_pedicel]="0.1"
rosparams[threshold_sepal]="0.2"

for i in "${!rosparams[@]}"
do
	echo "$i: ${rosparams[$i]}"
	rosparam set $i ${rosparams[$i]}
done




