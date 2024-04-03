# WaveformInversionUST
Frequency-Domain Waveform Inversion Ultrasound Tomography (UST) Using a Ring-Array Transducer

Ultrasound tomography (UST) is a medical imaging system that uses the transmission of ultrasound through tissue to create images of sound speed and attenuation.
The main application of UST is breast imaging, where images of sound speed and attenuation, alongside B-mode reflectivity imaging, are used to identify cancer. Currently, the main algorithm used to provide high-resolution images of sound speed and attenuation is waveform inversion. 

Our previous waveform inversion work ([rehmanali1994/FullWaveformInversionUSCT](https://github.com/rehmanali1994/FullWaveformInversionUSCT)) relies on the paraxial approximation to quickly model the plane wave propagation between two linear arrays in a pitch-catch setup. However, the paraxial approximation is no longer accurate for the diverging-wave single-element transmissions from a ring-array transducer. Because this work presents waveform inversion UST with a ring-array transducer, we must solve the full Helmholtz equation to accurately model wave propagation. A block LU approach is used to solve the Helmholtz equation efficiently on the GPU. See the full paper (citation below) for more details. 

<!---
Additionally, while our previous work was entirely based on simulated data, this work includes experimental data from a phantom, and _in-vivo_ breast data from an early prototype. As a result, this code is far more mature in terms of handling real experimental conditions (e.g., source estimation, gridding, windowing, outlier removal, etc.). We provide several experimental (_in-vitro_ and _in-vivo_) datasets that allow others in the UST community to test their algorithms against our algorithms on real (non-simulated) data.
--->

We show the waveform inversion UST reconstruction of both sound speed and attenuation in a phantom and _in-vivo_ breast imaging cases. The primary motivation of this open-source work is to demonstrate waveform inversion UST in a transparent manner that allows other researchers to easily reproduce our work and improve upon it. The sample data and algorithms provided in this respository were used in following work:

```BibTeX
@article{ali2024ringFWI2D,
  author={Ali, Rehman and Mitcham, Trevor M. and Brevett, Thurston and Agudo, Òscar Calderón and Martinez, Cristina Durán and Li, Cuiping and Doyley, Marvin M. and Duric, Nebojsa},
  journal={IEEE Transactions on Medical Imaging}, 
  title={2-D Slicewise Waveform Inversion of Sound Speed and Acoustic Attenuation for Ring Array Ultrasound Tomography Based on a Block LU Solver}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Attenuation;Mathematical models;Ultrasonic imaging;Tomography;Image reconstruction;Acoustics;Frequency-domain analysis;Frequency Domain;Waveform Inversion;Tomography;Ultrasound;Ring Array},
  doi={10.1109/TMI.2024.3383816}
}
```

If you use the algorithms and/or datasets provided in this repository for your own research work, please cite the above paper. You can find the Early Access version of the article [here](https://ieeexplore.ieee.org/document/10486959).

You can reference a static version of this code by its DOI number: [![DOI](https://zenodo.org/badge/684631232.svg)](https://zenodo.org/badge/latestdoi/684631232)

# Experimental Datasets

**Please download the sample data (BenignCyst.mat; Malignancy.mat; VSX_YezitronixPhantom1.mat; VSX_YezitronixPhantom2.mat) under the [releases](https://github.com/rehmanali1994/WaveformInversionUST/releases) tab for this repository, and place that data in the [SampleData](https://github.com/rehmanali1994/WaveformInversionUST/tree/main/SampleData/) folder.**

The following scripts correspond to each dataset:
1) BenignCyst.mat and Malignancy.mat - [MultiFrequencyWaveformInvKCI.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/MultiFrequencyWaveformInvKCI.m) - These are _in-vivo_ breast datasets corresponding to an imaging case with a benign cyst and an imaging case with a malignancy. These datasets were acquired at the Karmanos Cancer Institute (KCI) under IRB No. 040912M1F. In the full paper, we show results for (1024 emitters) x (1024 receivers), but to conserve space, we downsampled this data to (512 emitters) x (512 receivers). 
2) VSX_YezitronixPhantom1.mat and VSX_YezitronixPhantom2.mat - [MultiFrequencyWaveformInvVSX.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/MultiFrequencyWaveformInvVSX.m) - These datasets correspond to acquisitions from two different slices of a breast phantom using a Verasonics system. In the full paper, we show results for (1024 emitters) x (1024 receivers), but to conserve space, we downsampled this data to (256 emitters) x (256 receivers).

# k-Wave Simulations

In our past works, we included [k-Wave](http://www.k-wave.org/) simulated datasets. This time, rather than provide the datasets themselves, we provide the code to run the k-Wave simulations that generates those datasets. We do this to conserve space and provide codes to simulate UST using [k-Wave](http://www.k-wave.org/). Each [k-Wave](http://www.k-wave.org/) simulation involves a 3-step process:

1) [GenKWaveSimInfo.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Simulations/GenKWaveSimInfo.m) creates a MAT file that is stored in the [Simulations/sim_info](https://github.com/rehmanali1994/WaveformInversionUST/tree/main/Simulations/sim_info) folder. This MAT file contains all the information (medium, ring array geometry, and pulse excitation) needed to simulate the UST system. [GenKWaveSimInfo.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Simulations/GenKWaveSimInfo.m) uses [sampled_circle.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Simulations/phantoms/sampled_circle.m) to create the ring-array transducer and [soundSpeedPhantom2D.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Simulations/phantoms/soundSpeedPhantom2D.m) to produce the material properties used in each simulation. Two breast imaging cases are simulated: `option = 1` is from a breast CT; `option = 2` is from a breast MRI.
2) After generating the MAT file in [Simulations/sim_info](https://github.com/rehmanali1994/WaveformInversionUST/tree/main/Simulations/sim_info), we run the actual [k-Wave](http://www.k-wave.org/) simulation using [GenRFDataSingleTxKWave.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Simulations/GenRFDataSingleTxKWave.m). The `option` parameter corresponding to the simulation case must be specified. [GenRFDataSingleTxKWave.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Simulations/GenRFDataSingleTxKWave.m) loops through each single-element transmit. The simulated data for each transmit is then stored in MAT files the [Simulations/scratch](https://github.com/rehmanali1994/WaveformInversionUST/tree/main/Simulations/scratch) folder.
3) Lastly, [AssembleUSCTChannelData.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Simulations/AssembleUSCTChannelData.m) assembles the simulated data from each indivdual transmit/MAT-file in the [Simulations/scratch](https://github.com/rehmanali1994/WaveformInversionUST/tree/main/Simulations/scratch) folder into a single MAT file containing the full UST dataset in the [Simulations/datasets](https://github.com/rehmanali1994/WaveformInversionUST/tree/main/Simulations/datasets) folder: kWave_BreastCT.mat (`option = 1`); kWave_BreastMRI.mat (`option = 2`).

# Code

The key functions/classes used in the waveform inversion scripts ([MultiFrequencyWaveformInvKCI.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/MultiFrequencyWaveformInvKCI.m); [MultiFrequencyWaveformInvVSX.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/MultiFrequencyWaveformInvVSX.m); [MultiFrequencyWaveformInvkWave.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/MultiFrequencyWaveformInvkWave.m)) are: 
1) [HelmholtzSolver.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Functions/HelmholtzSolver.m) - Implements the Helmholtz equation solver as a class. For a given set of medium properties, the HelmholtzSolver forms the discretized system of equations that needs to be solved either on CPU or GPU. If an NVIDIA GPU is available, a block LU factorization is performed and stored in memory for subsequent solves using this factorization.
2) [stencilOptParams.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Functions/stencilOptParams.m) - Helper function called by [HelmholtzSolver.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Functions/HelmholtzSolver.m) to generate the stencil used to discretize the Helmholtz equation.
3) [decompBlockLU.cu](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Functions/decompBlockLU.cu) - This is the MEX CUDA code called by [HelmholtzSolver.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Functions/HelmholtzSolver.m) to perform the block LU factorization. Must be compiled in MATLAB using `mexcuda -lcusolver decompBlockLU.cu` using the cuSOLVER option.
4) [applyBlockLU.cu](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Functions/applyBlockLU.cu) - This is the MEX CUDA code called by [HelmholtzSolver.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Functions/HelmholtzSolver.m) to apply the block LU factorization computed by [decompBlockLU.cu](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Functions/decompBlockLU.cu) to set of given sources (or adjoint sources). Must be compiled in MATLAB using `mexcuda -lcublas applyBlockLU.cu` using the cuBLAS option.
5) [ringingRemovalFilt.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Functions/ringingRemovalFilt.m) - Helper function called during waveform inversion scripts ([MultiFrequencyWaveformInvKCI.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/MultiFrequencyWaveformInvKCI.m); [MultiFrequencyWaveformInvVSX.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/MultiFrequencyWaveformInvVSX.m); [MultiFrequencyWaveformInvkWave.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/MultiFrequencyWaveformInvkWave.m)) to remove ringing artifacts in the images.

These codes ran successfully with an NVIDIA GeForce RTX 3060 GPU (12 GB of GPU RAM) on a CPU with 40 GB of RAM in both MATLAB 2021b and 2022b. We therefore recommend running this code on a CPU with at least 32 GB of RAM and a GPU with at least 12 GB of RAM.

# Sample Results
Each waveform inversion script ([MultiFrequencyWaveformInvkWave.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/MultiFrequencyWaveformInvkWave.m); [MultiFrequencyWaveformInvKCI.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/MultiFrequencyWaveformInvKCI.m); [MultiFrequencyWaveformInvVSX.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/MultiFrequencyWaveformInvVSX.m)) saves the results at each iteration to a MAT file in the [Results](https://github.com/rehmanali1994/WaveformInversionUST/tree/main/Results) folder. The results stored in these MAT files can later be visualized using the [viewSavedResults.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/viewSavedResults.m)) script. 

1) BenignCyst.mat:

![](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Results/BenignCyst.gif)

2) Malignancy.mat

![](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Results/Malignancy.gif)

3) VSX_YezitronixPhantom1.mat

![](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Results/VSX_YezitronixPhantom1.gif)

4) VSX_YezitronixPhantom2.mat

![](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Results/VSX_YezitronixPhantom2.gif)

5) kWave_BreastCT.gif

![](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Results/kWave_BreastCT.gif)

6) kWave_BreastMRI.gif

![](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Results/kWave_BreastMRI.gif)
