# ECSE-316 Assignment 2
## Chrisovalantis Vatos (260989496)						 
## Sadek Mahmood (261053809)	

## Installation

Ensure Python 3.5+ is installed.
Dependencies used for this assignment include:
- numpy
- matplotlib
- opencv-python 
- pytest

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

The program is executed from the command line using the following syntax:

```bash
python fft.py [-m mode] [-i image]
```

**mode (optional):**
- [1] (Default) for fast mode where the image is converted into its FFT form and displayed
- [2] for denoising where the image is denoised by applying an FFT, truncating high
frequencies and then displayed
- [3] for compressing and plot the image
- [4] for plotting the runtime graphs for the report

**image (optional):**
- Filename of the image we wish to take the DFT of. (Default: the file name
of the image given to you for the assignment)

## Test Execution

The test suite is executed from the command line using the following syntax:

```bash
pytest -v
```


