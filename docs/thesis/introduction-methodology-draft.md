# Thesis draft: introduction and methodology

## 1. Introduction

Federated learning trains a shared model using data held by multiple clients. In federated averaging (FedAvg), a server sends the current model to the participating clients. Each client trains on its local examples and returns its updated model, and the server forms a weighted average. This arrangement keeps examples at the client during training, but it requires repeated exchanges of model-sized messages. The total communication depends on how many rounds are needed and on how much data is sent in each direction per round. [McMahan et al., 2017](https://proceedings.mlr.press/v54/mcmahan17a.html)

This thesis studies compression of client-to-server training updates. Quantized stochastic gradient descent (QSGD) discretizes a vector by stochastically rounding its normalized coordinates. For a given input vector, averaging over the rounding draws recovers that vector in expectation. Lossy Lempel-Ziv coding (LLZ) then compresses the integer sequence by exploiting repeated patterns and can allow approximate matches at a prescribed tolerance. In this thesis the input is a client's model change after local training. The experiments use FedAvg, so the update and aggregation protocol should be considered when interpreting comparisons with published gradient-compression results. [Alistarh et al., 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6c340f25839e6acdc73414517203f5f0-Abstract.html) · [Cao et al., 2026](https://doi.org/10.1109/TCOMM.2026.3745158)

The research question is whether selecting an LLZ tolerance separately for each client update reduces training upload size relative to fixed LLZ and QSGD while preserving model quality. The proposed selector starts with one QSGD draw and encodes the resulting sequence with several LLZ tolerances. It selects the smallest serialized reply that meets a bound on the additional reconstruction error from LLZ. The comparison tests whether those savings persist after accounting for message metadata and whether they justify the extra encoding work. Downlink size is recorded because a reduction in client uploads can account for only part of total communication.

The first study used DigiFace-1M, a synthetic face dataset, as a pilot to exercise the implementation and the budget calibration procedure. One of two subsequent confirmation seeds exceeded the predeclared one-percentage-point validation accuracy drop, so that pilot motivates testing the calibration procedure on a separate benchmark. The main benchmark uses CIFAR-10, a ten-class image classification dataset, and ResNet-18, a model and dataset pairing also used in the LLZ study. The scope is a four-client simulation with balanced partitions. Questions about device heterogeneity, privacy guarantees, and network performance are outside this study's scope. [Bae et al., 2023](https://openaccess.thecvf.com/content/WACV2023/html/Bae_DigiFace-1M_1_Million_Digital_Face_Images_for_Face_Recognition_WACV_2023_paper.html) · [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) · [Cao et al., 2026](https://doi.org/10.1109/TCOMM.2026.3745158)

## 2. Methodology

### 2.1 Data and model

The official CIFAR-10 training set contains 50,000 RGB images of size 32 by 32. The experiment makes a fixed, class-balanced split of 45,000 training and 5,000 validation images using split seed 42. Four simulated clients each receive 11,250 training and 1,250 validation examples with equal representation of the ten classes. The official 10,000 test images are reserved for evaluation after model and compressor selection. Images are scaled to the interval [-1, 1], and training uses random horizontal flips with probability 0.5. [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

The experiment trains ResNet-18 from random initialization. For 32-by-32 inputs, its initial convolution uses a 3-by-3 kernel with stride one and the initial max-pooling layer is removed. The model uses 32-group GroupNorm and a ten-class output layer. GroupNorm keeps the exchanged model state free of batch-normalization running statistics and integer batch counters. [He et al., 2016](https://arxiv.org/abs/1512.03385) · [Wu and He, 2018](https://arxiv.org/abs/1803.08494)

### 2.2 Federated training and model selection

Flower orchestrates the four simulated clients and PyTorch runs local training. Each round includes all four clients. Starting from the server state $w_t$, client $k$ runs one local epoch of stochastic gradient descent on the cross-entropy objective with a batch size of 64. In the initial baseline, the learning rate is 0.01, momentum and weight decay are zero, and training runs for 15 communication rounds. The server weights client contributions by their training sample counts, so each client contributes one quarter of the aggregate in this split. The uncompressed control averages the full local model states. In compressed runs, the client encodes the model delta $u_{k,t} = w_{k,t}^{\mathrm{local}} - w_t$. The server decodes each reply and adds the weighted mean of the reconstructed deltas to $w_t$.

Validation runs after each round on the fixed client validation splits. Checkpoint selection uses the lowest aggregate validation loss and favors the earlier round in a tie. The initial full run uses training seed 42 and provides a validation reference for tuning. The learning-rate pilot compares a rate of 0.03 with the first five rounds of the 0.01 reference using the same data split, initialization, and other training settings. The five-round comparison screens the candidate for a full run. The held-out test split stays reserved while the training configuration is being selected.

### 2.3 Adaptive LLZ

Each client applies QSGD once per floating-point tensor of its update and reuses the resulting signed integer codes and scales for each LLZ candidate. The project QSGD baseline uses a dense packet format that stores a code for every coordinate along with the reconstruction scale and packet metadata. All LLZ candidates for a given reply share the same QSGD realization. This controls for quantization randomness when comparing their sizes and reconstruction errors.

For a nonzero client update $u$, let $z$ be the reconstruction obtained with QSGD alone and $z_p$ the reconstruction after locally decoding LLZ with tolerance $p$. The selector admits a candidate when

$$
\frac{\|z_p-z\|_2^2}{\|u\|_2^2} \le \tau .
$$

The numerator measures the extra squared error introduced by LLZ relative to the QSGD reconstruction. The denominator scales it by the energy of the original update. The total squared error $\|z_p-u\|_2^2$ is measured separately. For a zero update the additional error must be zero. The project implementation considers $p \in \{0,1,2\}$. The lossless $p=0$ option preserves the QSGD reconstruction and is always feasible. The client chooses the feasible reply with the smallest serialized size, preferring lower $p$ when sizes tie.

CIFAR-10 budget calibration is planned after training settings are fixed. It will use saved training updates to examine candidate reply sizes and reconstruction errors. The subsequent learning runs will test whether the budget that controls update distortion also preserves validation accuracy. The DigiFace pilot budget is reported with its own experimental results.

### 2.4 Outcomes and communication measurement

The core comparison will include uncompressed FedAvg, QSGD, QSGD with lossless LLZ, and adaptive LLZ. Where settings are matched, changes relative to QSGD will be measured on the same training seeds and data partitions. Validation accuracy and loss track model quality. Client encoding and server decoding times indicate the computation cost of compression. Codec packet bytes capture the size of the encoded update, including packet headers. Training upload bytes capture the size of the full serialized client reply, including metadata and metrics. Total serialized bytes sum the training and validation messages in both directions.

The implementation measures the serialized Flower message and its unique descendant objects at the server's Grid interface, separately for each logical recipient. This boundary includes the model downloads used for training and validation. It excludes transport headers, acknowledgements, retransmissions, and other runtime traffic. Figures will label the numerator and denominator of each communication ratio so readers can distinguish a codec-size ratio from an experiment-wide communication reduction.
