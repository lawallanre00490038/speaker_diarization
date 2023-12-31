# Speaker Diarization Project

## Introduction

This project aims to perform speaker diarization on an audio file, which is the process of segmenting and labeling different speakers in an audio recording. It utilizes various tools and libraries to achieve this, including the Whisper ASR system, PyAnnote, and more.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running the Speaker Diarization](#running-the-speaker-diarization)
  - [Customizing the Model and Parameters](#customizing-the-model-and-parameters)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

### Prerequisites

Before running this project, ensure you have the following prerequisites installed:

- Python 3.x
- `pip` package manager
- Necessary dependencies, which can be installed using the provided `requirements.txt` file.

### Installation

1. Clone the repository to your local machine:

    git clone https://github.com/your-username/speaker_diarization.git

2. Change to the project directory:

    cd speaker_diarization

### Install the required Python dependencies:
pip install -r requirements.txt




### Running the Speaker Diarization
To perform speaker diarization on an audio file, use the following command:
    
    python speaker_diarization.py --audio_path audio_path.wav --num_speakers 2

The `--audio_path` argument specifies the path to the audio file to be processed, and the `--num_speakers` argument specifies the number of speakers to be detected in the audio file. The output will be saved to the `output` directory.

### Customizing the Model and Parameters
You can customize the model size by providing the --model_size parameter. Choose from 'small', 'medium', or 'large'.

### Results
Upon running the speaker diarization, the following results are generated:

### Speaker diarization labels for each segment in the audio.
A transcript file (transcript.txt) containing the transcribed speech with labeled speaker segments.
Visualizations of the speaker clusters using PCA in both 2D and 3D.
Troubleshooting
If you encounter any issues while running the project, consider the following:

### Ensure that all prerequisites are correctly installed.
Check that the audio file path provided is correct and the file exists.
Verify that the Whisper ASR system and its dependencies are functioning as expected.
Refer to the troubleshooting section in the documentation for any specific error messages.
Contributing
Contributions to this project are welcome. If you'd like to enhance or fix issues in the codebase, please follow these steps:

### Fork the repository.
Create a new branch for your feature or bug fix.
Make your changes and commit them.
Submit a pull request to the original repository.


### License
By Anthony and Olanrewaju