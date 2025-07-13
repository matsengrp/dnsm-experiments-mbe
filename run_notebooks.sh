#!/bin/bash

mkdir -p notebooks/_ignore
for nb in notebooks/*.ipynb; do
    output_notebook="notebooks/_ignore/$(basename "$nb")"
    if [ -f "$output_notebook" ]; then
	echo "Skipping $(basename "$nb") (already executed)."
	continue
    fi
    echo "Executing $(basename "$nb")..."
    echo "Output will be written to $output_notebook."
    echo "working directory: $(pwd)"
    jupyter nbconvert --to notebook --execute "$nb" --output-dir="." --output "$output_notebook" > $output_notebook.log 2>&1
    if [ $? -ne 0 ]; then
	echo "Error executing $(basename "$nb"), but continuing to the next notebook."
    fi
done
