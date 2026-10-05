#!/bin/sh
set -eu
# Download all Google Colab notebooks listed in README.

rm -f notebooks.txt *.ipynb

grep -o 'https://colab.research.google.com/drive/[0-9A-Za-z_-]*' ../README.md \
    | sed 's#https://colab.research.google.com/drive/##' \
    | sort -u > notebooks.txt

while IFS= read -r ID; do
    [ -n "$ID" ] || continue
    echo "Downloading notebook: $ID"
    wget --content-disposition \
        "https://drive.google.com/uc?export=download&id=$ID"
done < notebooks.txt

rm -f notebooks.txt
echo "All notebooks downloaded"
