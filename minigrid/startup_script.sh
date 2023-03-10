#!/bin/bash
echo 'Inside script'
if [ $# -eq 0 ]
  then
    /bin/bash
else
  $1
fi