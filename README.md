# Systematic Car Classification System
[![python](https://img.shields.io/badge/Python-3.x-ff69b4.svg)]()
[![tensorflow](https://img.shields.io/badge/Tensorflow-1.1x%7C2.0-brightgreen.svg)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-3.x%7C4.x-orange.svg)]()

## Introduction
This project is based on the architecture described in the Systematic Car Classification System (SCCS) paper.

>1. [Systematic Car Classification System (SCCS)](Systematic%20Car%20Classification%20System%20(SCCS).pdf)

## Requirements
  - "data" folder (downloaded from link in data.txt)
  - "models" folder (downloaded from link in data.txt)
  - "yolo" folder with the weights (downloaded from link in data.txt)
  - python
  - numpy
  - tensorflow
  - keras
  - pandas
  - opencv
  - scipy

## Usage

Use --help to see usage of main.py:
```
$ python main.py --image_path "data/test/bmw_x3_2014.jpg" --actual_label "BMW X3 2014"
```
```
$ python main.py [--help] [--actual_label ACTUAL_LABEL] [--image_path IMAGE_PATH] [--csv_path CSV_PATH]

required arguments:
  -l, --actual_label            label identifying the image


optional arguments:
  -i, --image_path              path to an input image
  -c, --csv_path                path to a .csv file containing data with the columns (filename, label)
```