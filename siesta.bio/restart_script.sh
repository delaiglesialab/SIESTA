#!/bin/bash

source /home/siesta/.venv/bin/activate

cd /home/siesta/UW-SIESTA

git pull origin main
git lfs pull

pkill -f 'celery worker'

./start_celery.sh

pa_reload_webapp siesta
