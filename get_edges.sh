#!/bin/bash
for image in dataset/*.jpg
do
  bn=$(basename $image .jpg)
  echo "Getting edges for $image"
  python canny_edge.py $image edges/$bn.jpg
done
