#!/usr/bin/env bash
# shellcheck source=~
. ~/venv/bin/activate
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
    [[ "${DEV}" != "-d" && -f wuhan${DEV}.log ]] && cat wuhan${DEV}.log >>wuhan${DEV}.old.log
    rm -f wuhan${DEV}.log
    nohup python3 wuhan.py ${DEV} 1>wuhan${DEV}.log 2>&1 &
    sleep 1
    python3 wuhan.py ${DEV} -s >/dev/null
    tail -f wuhan${DEV}.log
  }
else
  python3 wuhan.py ${DEV} "$@"
fi
