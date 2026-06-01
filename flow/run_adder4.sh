#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

rm -rf results/nangate45/adder4 logs/nangate45/adder4 objects/nangate45/adder4 reports/nangate45/adder4

util/docker_shell make DESIGN_CONFIG=/work/designs/nangate45/adder4/config.mk DESIGN_HOME=/work/designs
