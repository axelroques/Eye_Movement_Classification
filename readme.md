# Implementation of eye movement classification algorithm

**Based on _Komogortsev et. al. (2009)_.**

---

Every algorithm presented here can be described in the following general form.  
The input to an algorithm is provided as
a sequence of the eye-gaze position tuples $(t, x, y)$ where $x$ and $y$ are horizontal and vertical coordinates of the eye position sample, respectively, and $t$ is the timestamp of that sample. A threshold value is provided to allow classification of each eye position sample as a fixation or a saccade, according to the classification scheme implemented in each algorithm.  
Finally, a 'merge function' is employed to collapse successive samples of the same type into a single segment:

- Consecutive fixation samples are merged into larger fixation segments if two conditions are respected: the duration between two fixation segments should be less than $75 ms$ (filter for blinks) and the Euclidian distance between these fixation centroids should be less than $0.5$° (filter for microsaccades). The center of the resulting merged fixation segment is calculated as a the centroid of the various fixation segments. Note that fixations with duration less than $100 ms$ are discarded from the analysis.
- Consecutive samples that are classified as saccades are collapsed into a single saccade with onset and offset coordinates. Micro saccades with amplitudes of less than $0.5$° and saccades that contain eye positions not detected by the eye tracker are discarded from the analysis.

---

Currently, the following algorithms are implemented.

---

## Velocity Threshold Identification (IVT)

First described in _Salvucci and Goldberg (2000)_. Each sampled is classified as either a fixation or a saccade using a threshold value on the signal's velocity.

---

## Dispersion Threshold Identification (IDT)

First described in _Salvucci and Goldberg (2000)_. The algorithm defines a temporal window which moves one point at a time and the spacial dispersion created by the points within this window is compared against the threshold. If such dispersion is below the threshold, the points within the temporal window are classified as a part of fixation; otherwise, the window is moved by one sample, and the first sample of the previous window is classified as a saccade.

---

## Minimum Spanning Tree Identification (MST)

First described in _Salvucci and Goldberg (2000)_. The I-MST is a dispersion-based identification algorithm that builds a minimum spanning tree taking a predefined number of eye position points using Prim's algorithm. The I-MST traverses the MST and separates the points into fixations and saccades based on the predefined distance thresholds. The I-MST requires a sampling window to build a sequence of MST trees allowing it to parse a long eye movement recording. Here, the window selected is 200ms.

---

## Hidden Markov Model Identification (IHMM)

First described in _Salvucci and Goldberg (2000)_. The first stage of the I-HMM is identical to I-VT, where each eye position sample is classified either as a fixation or a saccade depending on the velocity threshold. Second stage is defined by the Viterbi Sampler (_Forney (1973)_), where each eye position can be re-classified, depending on the probabilistic parameters (initial state, state transition and observation probability distributions) of the model. The goal of the Viterbi Sampler is to maximize the probability of the state assignment given probabilistic parameters of the model. The initial probabilistic parameters given to I-HMM are not optimal and can be improved. The third stage of the I-HMM is defined by Baum-Welch re-estimation algorithm (_Baum et al. (1970)_). This algorithm re-estimates initial probabilistic parameters and attempts to minimize errors in the state assignments. Parameter re-estimation performed by Baum-Welch can be conducted multiple times. Here, the number of such re-estimations is four.

## Kalman Filter Identification (IKF)

Here, we employ a Two State Kalman Filter (TSKF), first described in _Komogortsev and Khan (2009)_. The TSKF models an eye as a system with two states: position and velocity. The acceleration of the eye is modeled as white noise with fixed maximum acceleration. When applied to the recorded eye position signal the TSKF generates predicted eye velocity signal. The values of the measured and predicted eye velocity allow employing Chi-square test to detect the onset and the offset of a saccade (_Sauter (1991)_).

---

## Identification by two-means clustering (I2MC)

Described in _Hessels et. al. (2016)_. It is a fixation detection algorithm suited for a wide range of noise levels and when periods of data loss may be present, without the need to set a large number of parameters.
The I2MC algorithm is composed of three separate steps:

- Interpolation of missing data.
- Two-means clustering (i.e., the selection of fixation candidates by the search rule).
- Fixation labeling (the categorization rules).

**Interpolation**. Steffen's method is used (Steffen, 1990) if the segment with missing data is shorter than 100ms This value is not fixed and is a parameter of this model. Interpolation is monotonic and locally determined by the valid gaze samples at the beginning and the end of the interpolation window. Furthermore, at least two valid points of data must be present at the beginning and at the end of the interpolation window.

**Two means clustering**. A moving window of 200 ms width slides over the gaze position signal. The value of 200 ms was chosen here so that a window generally would contain parts of at most two, and no more, fixations. For each window, a two-means clustering procedure is carried out. Two-means cluster-ing is a variant of k-means clustering where $k = 2$.

**Fixation labelling**. A cutoff is used to determine fixation candidates from the clustering-weight signal. Here we used a cutoff of the mean clustering weight plus two standard deviations. All periods of clustering-weight signal below this cutoff are labeled as fixation candidates, and thereafter consecutive fixation candidates are merged. Finally, short fixation candidates are excluded from the output. Standard values are: merging fixation candidates that were less than $0.7^o$ apart and separated by less than $40 ms$and removing fixations shorter than $40 ms$ were removed.

---

## Naive Segmented Linear Regression and Hidden Markov Models (NSLR-HMM)

Described in _Pekkanen & Lappi (2017)_. It is a classification algorithm for the detection of fixations, saccades, smooth pursuits and post-saccadic oscillations.

This algorithm introduces the Naive Segmented Linear Regression (NSLR), a new method for eye-movement signal denoising and segmentation. The proposed method is based on the assumption that in most situations the gaze position signal is well approximated by a piecewise linear function, and events correspond to the pieces of such function. The method finds a piecewise linear function that approximately minimizes the approximation error, while taking into account prior knowledge of typical eye movement characteristics.

A full Python implementation of the method is available under an open source license at _https://gitlab.com/nslr/_. This implementation is heavily based on this repository.
