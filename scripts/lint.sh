#!/bin/bash
set -x

# shellcheck disable=SC2046
base_dir=$(dirname $(dirname $0))

isort --sp "${base_dir}/pyproject.toml" --check .

black --config "${base_dir}/pyproject.toml" --check .

flake8 --config "${base_dir}/setup.cfg" .
