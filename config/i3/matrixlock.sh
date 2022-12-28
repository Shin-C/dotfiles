#!/bin/bash

i3-sensible-terminal -e cmatrix &
sleep 0.2

i3-msg fullscreen toggle global
i3lock -n; i3-msg kill
