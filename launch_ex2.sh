#!/bin/bash

python server_main.py --username root --password linux123 --port 3001 --world_size 2 --round 2 --offline-epoch 2&

python client_main.py --username root --password linux123 --port 3001 --world_size 2 --rank 1 &

wait