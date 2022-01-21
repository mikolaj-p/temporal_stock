# temporal_stock

This repository contains code and dataset for a simple experiment using temporal networks approach to analyzing stock market data.

The main script is based on pandas, networkx and matplotlib libraries. All the requirements are described in requirements.txt.
It is strongly suggested to use virtual environment to install all the required libraries.

Run temporal_stock.py without any parameters to run the default setup. The script will store plots of the statistics in the current working directory.

# Additional parameters:
* delta - size of the window used in Pearson's correlation coefficients calculation, default: 300,
* shift - shift size used to generate all the graphs, default: 25,
* min - compute minimal spanning tree instead of maximal spanning tree, default: maximal spanning tree,
* data-dir - path to the directory containing the data, default: stock,
* instr-max - use only that number of instruments, for testing purposes, default: 470 (all the instruments in stock directory),
* plot-graphs - plot graph for each step, default: false.

