#!/bin/bash

# Script used to start the single job server service. First four lines are copied
# from the PYNQ jupyter notebook startup script, to source the relevant environment variables, etc

# Source the environment as the init system won't
set -a
. /etc/environment
set +a
for f in /etc/profile.d/*.sh; do source $f; done

/usr/local/share/pynq-venv/bin/python /home/xilinx/software/qubic/soc_rpc_server.py /home/xilinx/server_config.yaml
