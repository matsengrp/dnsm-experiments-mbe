#!/bin/bash
rsync -avz --delete quokka:/home/matsen/re/dnsm-experiments-1/dnsm-train/_logs .

find _logs/* -type d | parallel -j 8 python ../scripts/renumber_logs.py {}
