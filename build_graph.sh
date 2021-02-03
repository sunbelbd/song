#!/bin/bash

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <build_data> <row> <dimension> <l2/ip/cos>" >&2
  echo "For example: $0 letter.scale 15000 26 cos" >&2
  exit 1
fi

$(dirname $0)/song build $1 0 0 $2 $3 0 $4
