# CompressorClassification.jl
## KNN-Gzip Parameter-Free Text Classification

Based on the paper [Less is More: Parameter-Free Text Classification with Gzip](https://arxiv.org/abs/2212.09410) (2023).
The original source code is at https://github.com/bazingagin/npc_gzip/.

This also includes tie breaking recommendations from Ken Schutte. See his blog post at https://kenschutte.com/gzip-knn-paper/.

## Method

The input is test data, reference data and reference labels. 
This method compresses each sample of the test data, each sample of the reference data and these two joined together: `test_data[j] * " " * ref_data[i]`.
The main idea is that the number of bytes in the compressed joined text is lowest for data that is similar.
The normalised compression distance (NCD) measures this with respect to the original samples:

```
ncd = (length_12 - min(length_1, length_2)) / max(length_1, length_2)
```

where `length_12` is the length of the compressed joined text and `length_1` and `length_2` are the lengths of the invididual compressed samples.

The K-nearest neighbour (KNN) algorithm is used to choose a label from the reference data for the test sample based on the NCDs. 
That is, the most common label amongst the K reference samples with the lowest NCDs is chosen. 
A problem with this is that there can be ties for the most common label. 
By default a single label is selected randomly. See [Tie breaking](#tie-breaking) for more detail.

## Usage

Calculate a label from the raw inputs:

```Julia
label = knn_classification(text, ref_data, labels; k=10)
```

It is highly recommend to calculate the distances first because this is a very slow and computationally intensive task.
Create a script called `calculate_distance_matrix.jl` which includes the following:

```Julia
using Arrow
using Arrow.Tables
distances = make_distance_matrix(test_data, ref_data)
Arrow.write(filepath, Tables.table(distance_matrix))
```

Run it with multiple threads using: `julia --threads auto calculate_distance_matrix.jl`.
As a baseline, creating a 7,200&times;120,000 matrix for the [AG News](https://huggingface.co/datasets/ag_news) dataset takes 3 hours on a 2.30 GHz processor with 16 threads with 16GB of RAM. The  Arrow file has a size of 6.8 GB.

Load the data with: 

```Julia
using Arrow
using Arrow.Tables
distances = Tables.matrix(Arrow.Table(filepath))
```

Use it as follows:

```Julia
label = knn_classification(distances[:, j], labels; k=10)
```

### Tie breaking

The original paper did not break ties.
To recreate the results in the original paper, use `knn_classification_multi` with `k=2` and the following accuracy function:

```Julia
test_acc = count(yy -> yy[2] in yy[1], zip(y_pred, y_test)) / length(y_test)
```

The method `knn_classification_multi` returns a vector so for type safety it is a different function to `knn_classification`.

It is highly recommended to use a tie breaking strategy. Three have been implemented:
- `:random` (default): randomly select a label with uniform probability.
- `:decrement`: decrement `k` and recalculate scores until the tie is broken or `k=1`.
- `:min_total`: select the most common class with the lowest total NCD. 
Another way to think of this is with KNN each reference sample has a vote with `weight=1`. For this method, each sample has a vote with `weight=NCD`.
This will align with the decrement strategy in most cases. See the tests for examples where it does not.

The accuracy function should be:
```Julia
test_acc = count(yy -> yy[2] == yy[1], zip(y_pred, y_test)) / length(y_test)
```

## Install

## Installation

Download the GitHub repository (it is not registered). Then in the Julia REPL:
```
julia> ] # enter package mode
(@v1.x) pkg> dev path\\to\\CompressorClassification.jl
julia> using Revise # for dynamic editing of code
julia> using CompressorClassification
```

Done. 
