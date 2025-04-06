#!/bin/bash

SESSION="wanshan"

# tmux new-session -d -s $SESSION

# tmux split-window -h -t $SESSION:0

# Wait a bit to ensure tmux initializes the session and panes

sleep 1

# Clear both panes before running commands
tmux send-keys -t $SESSION:1.0 "clear" C-m
tmux send-keys -t $SESSION:1.1 "clear" C-m

sleep 0.5

tmux send-keys -t $SESSION:1.0 "CUDA_VISIBLE_DEVICES=2 python inference_speed_streamer.py --use_cache --uigraph" C-m
tmux send-keys -t $SESSION:1.1 "CUDA_VISIBLE_DEVICES=3 python inference_speed_streamer.py --use_cache" C-m
