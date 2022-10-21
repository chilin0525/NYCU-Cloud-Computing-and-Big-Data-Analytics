# HW1

## Prerequisite

* unzip source ```ccbda-2022-hw1.zip```
    ```
    .
    ├── 43.hdf5             <=== model weight
    ├── ccbda-2022-hw1.zip  <=== unzip
    ├── inference.py
    ├── main.py
    ├── README.md
    ├── requirements.txt
    ├── result.csv
    └── video_ar.py
    ```
* predict your final file structure
    ```
    .
    ├── 43.hdf5
    ├── ccbda-2022-hw1.zip  
    ├── inference.py
    ├── main.py
    ├── README.md
    ├── requirements.txt
    ├── result.csv
    ├── train               <=== add
    ├── test                <=== add
    └── video_ar.py
    ```
* install package
    ```
    pip3 install -r requirements.txt 
    ```
* video argumentation for each video
    * double number of original video
    ```
    $ python3 video_ar.py
    ```

## Training

```
$ python3 main.py
```

## Inference

* generate result csv: ```result.csv```
    ```
    $ python3 inference.py
    ```