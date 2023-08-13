module CompressorClassification

using CodecZlib
using StatsBase
using ProgressMeter

export knn_classification, knn_classification_multi
export normalised_compression_distance
export make_distance_vector, make_distance_matrix

"""
    knn_classification(distance_vector, labels; k=2, tie_break=:random)
    knn_classification(text, ref_data, labels; k=2, tie_break=:random)

The most common class between the k-nearest neighbours according to the distance vector. 

Tie breaking strategies:
- `:random` (default): randomly select a label with uniform probability.
- `:decrement`: decrement `k` and recalculate scores until the tie is broken or `k=1`.
- `:min_total`: select the most common class with the lowest total NCD. 

Reference: [Less is More: Parameter-Free Text Classification with Gzip](https://arxiv.org/abs/2212.09410) (2023).
"""
function knn_classification(distances::Vector{Float64}, labels::Vector{T}; k::Int=2, tie_break::Symbol=:random) where T
    sorted_indices = sortperm(distances)
    top_labels = labels[sorted_indices[1:k]]
    most_common_classes = get_most_common_neighbours(top_labels)
    if length(most_common_classes) == 1
        predicted_class = most_common_classes[1]
    elseif tie_break == :random
        predicted_class = rand(most_common_classes)
    elseif tie_break == :decrement
        predicted_class = decrement_tie_break(top_labels, k)
    elseif tie_break == :min_total
        top_distances =  distances[sorted_indices[1:k]]
        predicted_class = find_min_total_label(top_labels, top_distances, most_common_classes) 
    else
        throw("Tie break strategy \"$tie_break\" is not known. Please choose: :random, :decrement or :min_total.")
    end
    predicted_class
end

function knn_classification(text::AbstractString, ref_data::Vector{<:AbstractString}, labels::Vector; options...)
    distances = make_distance_vector(text, ref_data)
    knn_classification(distances, labels; options...)
end

function get_most_common_neighbours(labels::Vector{T}) where T
    frequencies = countmap(labels)
    max_freq = maximum(values(frequencies)) 
    collect(keys(filter(pair->pair[2] == max_freq, frequencies)))
end

function decrement_tie_break(sorted_labels::Vector{T}, k::Int) where T
    kmod = k
    most_common_classes = get_most_common_neighbours(sorted_labels)
    while (length(most_common_classes) > 1)
        kmod = kmod - 1
        sorted_labels = sorted_labels[1:kmod]
        most_common_classes = get_most_common_neighbours(sorted_labels)
    end
    most_common_classes[1]
end

function find_min_total_label(labels::Vector{T}, distances::Vector{Float64}, selected_classes::Vector{T}) where T
    selected_idxs = [idx for idx in eachindex(labels) if (labels[idx] in selected_classes)]
    totals = calc_class_totals(distances[selected_idxs], labels[selected_idxs])
    pair = findmin(totals) # assumes that there are no further ties
    pair[2]
end

function calc_class_totals(distances::Vector{Float64}, labels::Vector{T}) where T
    totals = Dict{T, Float64}()
    for (d, l) in zip(distances, labels)
        if haskey(totals, l)
            totals[l] += d
        else
            totals[l] = d
        end
    end
    totals
end

"""
    knn_classification_multi(distance_vector, labels; k=2, tie_break=:random)
    knn_classification_multi(text, ref_data, labels; k=2, tie_break=:random)

The most common class between the k-nearest neighbours according to the distance vector.
Ties are not broken and all labels with the maximum count will be returned. 
This is closest to the method used in the original code for the reference paper.

Reference: [Less is More: Parameter-Free Text Classification with Gzip](https://arxiv.org/abs/2212.09410) (2023).
"""
function knn_classification_multi(distances::Vector{Float64}, labels::Vector{T}; k::Int=2) where T
    sorted_indices = sortperm(distances)
    top_labels = labels[sorted_indices[1:k]]
    most_common_classes = get_most_common_neighbours(top_labels)
    most_common_classes
end

function knn_classification_multi(text::AbstractString, ref_data::Vector{<:AbstractString}, labels::Vector; options...)
    distances = make_distance_vector(text, ref_data)
    knn_classification_multi(distances, labels; options...)
end

"""
    normalised_compression_distance(text1[, length1], text2 [, length2])

Returns the normalised compression distance between `text1` and `text2`.
"""
function normalised_compression_distance(text1::AbstractString, text2::AbstractString)
    compressed1 = transcode(GzipCompressor, text1)
    compressed2 = transcode(GzipCompressor, text2)
    compressed12 = transcode(GzipCompressor, text1 * " " * text2)

    length1 = length(compressed1)
    length2 = length(compressed2)
    length12 = length(compressed12)
    (length12 - min(length1, length2)) / max(length1, length2)
end

function normalised_compression_distance(text1::AbstractString, length1::Int, text2::AbstractString)
    compressed2 = transcode(GzipCompressor, text2)
    compressed12 = transcode(GzipCompressor, text1 * " " * text2)

    length2 = length(compressed2)
    length12 = length(compressed12)
    (length12 - min(length1, length2)) / max(length1, length2)
end

function normalised_compression_distance(text1::AbstractString, length1::Int, text2::AbstractString, length2::Int)
    compressed12 = transcode(GzipCompressor, text1 * " " * text2)
    length12 = length(compressed12)
    (length12 - min(length1, length2)) / max(length1, length2)
end

"""
    make_distance_matrix(test_data, ref_data)

Calculates the distance matrix based on the normalised compression distance.
Each column `j` is a distance vector for `test_data[j]`.
"""
function make_distance_matrix(test_data::AbstractVector{<:AbstractString}, ref_data::AbstractVector{<:AbstractString})
    nref = length(ref_data)
    ntest = length(test_data)
    lengths_ref = Vector{Int}(undef, nref)
    progress = Progress(nref, desc="ref data: ")
    Threads.@threads for i in 1:nref
        next!(progress)
        compressed2 = transcode(GzipCompressor, ref_data[i])
        lengths_ref[i] = length(compressed2)
    end
    finish!(progress)
    distances = Matrix{Float64}(undef, nref, ntest)
    progress = Progress(ntest, desc="test data:")
    Threads.@threads for j in 1:ntest
        compressed1 = transcode(GzipCompressor, test_data[j])
        length1 = length(compressed1)
        for i in 1:nref
            distance = normalised_compression_distance(test_data[j], length1, ref_data[i], lengths_ref[i])
            distances[i, j] = distance
        end
        next!(progress)
    end
    finish!(progress)
    distances
end

"""
    make_distance_vector(text, ref_data)

Calculates the distance vector based on the normalised compression distance.
"""
function make_distance_vector(text::AbstractString, ref_data::AbstractVector{<:AbstractString})
    nref = length(ref_data)
    distances = Vector{Float64}(undef, nref)
    compressed1 = transcode(GzipCompressor, text)
    length1 = length(compressed1)
    progress = Progress(nref)
    Threads.@threads for i in 1:nref
        distance = normalised_compression_distance(text, length1, ref_data[i])
        distances[i] = distance
        next!(progress)
    end
    finish!(progress)
    distances
end

end # module CompressorClassification