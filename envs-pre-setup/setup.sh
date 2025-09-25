#!/bin/bash
# Install SC2 and add the custom maps

if [ -z "$EXP_DIR" ]
then
    EXP_DIR=~
fi

echo "EXP_DIR: $EXP_DIR"
mkdir pymarl
cd $EXP_DIR/pymarl

mkdir 3rdparty
cd 3rdparty

export SC2PATH=`pwd`'/StarCraftII'
echo 'SC2PATH is set to '$SC2PATH

if [ ! -d $SC2PATH ]; then
        echo 'StarCraftII is not installed. Installing now ...';
        wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
        unzip -P iagreetotheeula SC2.4.10.zip
        rm -rf SC2.4.10.zip
else
        echo 'StarCraftII is already installed.'
fi

echo 'Adding SMAC maps.'
MAP_DIR="$SC2PATH/Maps/"
echo 'MAP_DIR is set to '$MAP_DIR

if [ ! -d $MAP_DIR ]; then
        mkdir -p $MAP_DIR
fi

cd ..
wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
unzip SMAC_Maps.zip
mv SMAC_Maps $MAP_DIR
rm -rf SMAC_Maps.zip

echo 'StarCraft II and SMAC are installed.'


#We make a clone for v2 smac to avoid conflict

# Make sure SC2PATH is set
if [ -z "$SC2PATH" ]; then
    echo "Error: SC2PATH environment variable is not set."
    exit 1
fi

MAPS_DIR="$SC2PATH/Maps"
echo "Using maps directory: $MAPS_DIR"

# Loop over all .SC2Map files
for file in "$MAPS_DIR"/*.SC2Map; do
    if [ -f "$file" ]; then
        if [[ "$file" == *_v2.SC2Map ]]; then
            echo "Skipping already converted file: $file"
            continue
        fi
        base=$(basename "$file" .SC2Map)
        new_file="$MAPS_DIR/${base}_v2.SC2Map"
        cp "$file" "$new_file"
        echo "Copied $file -> $new_file"
    fi
done

echo "All maps copied with _v2 suffix."


(cd football && pip install .)
# (cd smacv2 && pip install .) # v2 still need fixing
(cd smac && pip install .)
python test_parallel_api.py > output.log 2>&1