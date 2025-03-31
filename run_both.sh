#!/bin/bash

# Start a new tmux session named "mysession" in detached mode
# tmux new-session -d -s run_both 
# Send the first command to pane 0
tmux send-keys -t wanshan:1.0 'CUDA_VISIBLE_DEVICES=2 python inference.py --use_cache --uigraph' C-m

# Split horizontally, creating pane 1, and run the second command
tmux split-window -h -t mysession:0
tmux send-keys -t wanshan:1.1 'CUDA_VISIBLE_DEVICES=3 python inference.py --use_cache' C-m

# Attach to the tmux session so you can see both commands running
tmux attach -t wanshan
