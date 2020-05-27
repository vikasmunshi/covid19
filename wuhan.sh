#!/usr/bin/env bash
cd "$(dirname "$0")" || exit
if [[ $# -eq 0 ]]
then
  nohup python wuhan.py &
  python wuhan.py -s
else
  python wuhan.py "$@"
fi
