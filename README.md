# itDAS - The Iterative Delay-and-Sum Beamformer

itDAS is an iterative image reconstruction algorithm for use in breast
microwave imaging, based on the delay-and-sum beamformer [1]. 

This GitHub project page contains the implementation of the beamformer used
in its first publication [2] and all the code used to produce the results
presented in [2].

The data used to produce the results presented in [2] are available
[here](https://bit.ly/itDAS-data). To use these data files with this 
repository, place the data files into the `../data/` directory.

The .stl files of the 3D-printable phantoms used in this investigation
are available [here](https://bit.ly/itDAS-phantoms).

1. Hagness, S.C; Taflove, A.; Bridges, J. Two-dimensional FDTD analysis of a
pulsed microwave confocal system for breast cancer detection: fixed-focus
and antenna-array sensors. _IEEE Trans. Biomed. Eng._ **1998**, _45_,
1470-1479. DOI:[10.1109/10.730440](https://doi.org/10.1109/10.730440). 

2. Reimer, T.; Solis-Nepote, M.; Pistorius, S. The application of an iterative
structure to the delay-and-sum and the delay-multiply-and-sum beamformers in
breast microwave imaging. _Diagnostics_, submitted. 

## Getting Started

### Prerequisites

The Python requirements are:

- Python >=3.6
- Libraries in the `requirements.txt` file
    - numpy >= 1.16.2
    - pathlib >= 1.01
    - scipy >= 1.2.1
    - matplotlib >= 3.0.3

### Installing

We recommend using the Anaconda distribution for Python 3.x, which can be
 downloaded [here](https://www.anaconda.com/distribution/).
 
After installing a Python distribution, the required libraries can be
installed via the command line. After navigating to the project directory, 
enter the command:
 
```
pip install -r requirements.txt
```

This will install all the libraries listed in the `requirements.txt` file.

## Usage

### Running the Tests

The `../tests/` folder contains two Python test files and one 
Matlab/Octave test file. 

- The `../tests/check_requirements.py` file checks if the required libraries
 are installed.

### Using the Reconstruction Code

The code used to produce the reconstructions is contained in the `../umbms/`
directory. While the code in this repository is specific to the pre-clinical
imaging system at the University of Manitoba, the code can be adapted for use
with other scan systems. 

## Contributing

Please read the `CONTRIBUTING.md` file for details on contributing to the
project.


## Authors

- Tyson Reimer<sup>1</sup>
- Mario Solis-Nepote<sup>2</sup>
- Dr. Stephen Pistorius<sup>1,2</sup>

1. Department of Physics and Astronomy, University of Manitoba, Winnipeg,
Manitoba, Canada
2. Research Institute in Oncology and Hematology, University of Manitoba,
Winnipeg, Manitoba, Canada

## License

This project is licensed under the Apache 2.0 License. See the `LICENSE` file
for more information.
