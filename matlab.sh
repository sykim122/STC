#!/bin/sh

nohup /opt/local/R2015a/bin/matlab < $1 1> $1.out 2>$1.err &

exit
