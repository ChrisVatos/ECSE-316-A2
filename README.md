# ECSE-316 Assignment 2
## Chrisovalantis Vatos (260989496)						 
## Sadek Mahmood (261053809)	

## Installing Dependencies

Ensure Python 3.5+ is installed. The dependencies used for this assignment include:
- numpy
- matplotlib
- opencv-python 
- pytest

Install dependencies using the following command line call:

```bash
pip install -r requirements.txt
```

## Usage

The program is executed from the command line using the following syntax:

```bash
python fft.py [-m mode] [-i image]
```

**mode (optional):**
- [1] (Default) FFT + Display
- [2] Denoising + Display
- [3] Compression + Display
- [4] Plotting Runtimes

**image (optional):**
- Filename (path) of the image we wish to take the DFT of
- Default is ./assets/moonlanding.png

## Test Execution

The test suite is executed from the command line using the following:

```bash
pytest -v
```


