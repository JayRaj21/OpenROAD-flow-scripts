#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

util/docker_shell make DESIGN_CONFIG=/work/designs/nangate45/adder4/config.mk DESIGN_HOME=/work/designs lvs
