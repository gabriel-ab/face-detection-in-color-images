# Face Detection in Color Images
Paper implementation for face detection more information about technique in the paper: https://ieeexplore.ieee.org/document/1000242


## How to Execute

First, create an python virtual environment and install depedencies:

```sh
conda create -n face-detection python=3.11
conda activate face-detection
pip install -r requirements.txt
```


Then Run the script inside the package with:

```sh
python -m face_detection
```
    

If you want to see a working example from the paper, run:

```sh
python -m face_detection --example
```

## Notebook with step by step of code

You can read in detail step by step for the image processing in the [notebook](/notebook.ipynb)