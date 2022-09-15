#!/bin/bash

declare -A rosparams

# most-used
rosparams[pedicel_calc_mode]=4
rosparams[which_pedicel]=0

# less-used
rosparams[pedicel_cut_prop]="0.5"
rosparams[bbox_top]="0.5"
rosparams[ripeness_threshold]=10
rosparams[ripeness_percentile]="0.25"
rosparams[deg]=4

# instance segmentation thresholds
rosparams[threshold_stem]="0.3"
rosparams[threshold_tomato]="0.1"
rosparams[threshold_pedicel]="0.1"
rosparams[threshold_sepal]="0.2"

for i in "${!rosparams[@]}"
do
	rosparam set $i ${rosparams[$i]}
done




