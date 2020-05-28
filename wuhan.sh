#!/usr/bin/env bash
cd "$(dirname "$0")" || exit
for arg; do
  shift
  if [[ "$arg" == "-d" ]]; then
    DEV="-d"
  else
    set -- "$@" "$arg"
  fi
done
if [[ $# -eq 0 ]]; then
  python3 wuhan.py ${DEV} -s || {
    echo 'starting server ...'
    nohup python3 wuhan.py ${DEV} 1>wuhan${DEV}.log 2>&1 &
    python3 wuhan.py ${DEV} -s >/dev/null
    sleep 1
    python3 wuhan.py ${DEV} -s
  }
else
  python3 wuhan.py ${DEV} "$@"
fi
