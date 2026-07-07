#!/bin/bash

echo "Installing dependencies"

apt update

apt install gettext -y

cat '/dep/packages.txt' | envsubst | xargs apt -y install

rosdep update

rosdep install --from-paths /app --ignore-src  -r -y -q