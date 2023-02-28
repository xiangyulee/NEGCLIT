#!/bin/bash

python server_main.py --ip 127.0.0.1 --port 3001 --world_size 2 --round 2 &

python client_main.py --ip 127.0.0.1 --port 3001 --world_size 2 --rank 1 &

wait