# Summary.
python summary/run.py \
    -d /mnt/raid/nia_final/videos \
    -v /mnt/raid/nia_final/kbs.h5 \
    -s splits/split_1.json

# Description.
python description/run.py \
    -d /mnt/raid/nia_final/videos \
    -v /mnt/raid/nia_final/kbs.h5 \
    -s splits/split_1.json \
    -bs 128
