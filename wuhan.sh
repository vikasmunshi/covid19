#!/usr/bin/env bash
cd "$(dirname "$0")" || exit

for arg; do
  shift
  if [[ "$arg" == "-d" ]]; then
    DEV="-d"
  elif [[ "$arg" == "restart" ]]; then
    RESTART="restart"
  else
    set -- "$@" "$arg"
  fi
done

if [[ $# -eq 0 ]]; then
  [[ "$RESTART" == "restart" ]] && python3 wuhan.py ${DEV} -k
  python3 wuhan.py ${DEV} -s || {
    echo 'starting server ...'
    nohup python3 wuhan.py ${DEV} 1>wuhan${DEV}.log 2>&1 &
    sleep 1
    python3 wuhan.py ${DEV} -s >/dev/null
    if [[ "${DEV}" == "-d" ]]; then
      tail -f wuhan${DEV}.log
    else
      sleep 1
      python3 wuhan.py ${DEV} -s
    fi
  }
else
  python3 wuhan.py ${DEV} "$@"
fi
