#!/bin/bash

python3 sampler.py

if [ $? -ne 0 ]; then
  echo "sampler.py failed to run"
  exit 1
fi

python3 mfcc_len.py

if [ $? -ne 0 ]; then
  echo "mfcc_len.py failed to run"
  exit 1
fi

echo "sampled sucessfully"