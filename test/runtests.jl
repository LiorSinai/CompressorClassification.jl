using Test
using CompressorClassification
using StatsBase
using CodecZlib

test_data = [
    "Japan's Seiko Epson Corp. said Wednesday it has developed a 12-gram flying microrobot, the world's lightest.",
    "Michael Phelps won the gold medal in the 400 individual medley and set a  world record in a time of 4 minutes 8.26 seconds.",
    "Elliptic Curves Yield Their Secrets in a New Number System. Ana Caraiani and James Newton have extended an important result in number theory to the imaginary realm.",
]

ref_data = [
    "The latest tiny flying robot that could help in search and rescue or surveillance has been unveiled in Japan.",
    "Phelps adds yet another gold: Michael Phelps collected his fifth gold medal of the Games with victory in the men's 100 metres butterfly today.",
    "South Africa is a country on the southernmost tip of the African continent, marked by several distinct ecosystems.",
    "Neutrinos offer a new view of the Milky Way. Physicists used the ghostly subatomic particles coming from within our galaxy to make a new map.",
    "Lionel Messi makes it official and signs with Inter Miami. Lionel Messi has finalized his deal to join Major League Soccer, and after years of planning and pursuing, Inter Miami has landed a global icon.",
]

classes_int = [
    3,
    1,
    2,
    3,
    1,
]

classes_sym = [
    :science,
    :sports,
    :world,
    :science,
    :sport,
]

@testset "unit tests" begin 
    distance_similar = normalised_compression_distance(test_data[1], ref_data[1])
    @test distance_similar ≈ 0.6347826086956522

    length1 = length(transcode(GzipCompressor, test_data[1]))
    distance_similar_2 = normalised_compression_distance(test_data[1], length1, ref_data[1])
    @test distance_similar === distance_similar_2

    distance_diff = normalised_compression_distance(test_data[1], ref_data[4])
    @test distance_diff ≈ 0.6850393700787402

    distances = make_distance_vector(test_data[1], ref_data)
    expected_vector = [
        0.6347826086956522,
        0.6589147286821705,
        0.6434782608695652,
        0.6850393700787402,
        0.7142857142857143,
    ]
    @test distances ≈ expected_vector

    distance_matrix = make_distance_matrix(test_data, ref_data)
    @test size(distance_matrix) == (5, 3)
    @test distance_matrix[:, 1] == distances
end

@testset "kNN classification - top 1" begin 
    text = test_data[1]
    label_int = knn_classification(text, ref_data, classes_int; k=1)
    @test label_int == 3

    label_sym = knn_classification(text, ref_data, classes_sym; k=1)
    @test label_sym == :science

    distances = make_distance_vector(text, ref_data)
    label_int = knn_classification(distances, classes_int; k=1)
    @test label_int == 3
end

@testset "kNN classification - k=2 random" begin 
    ntrials = 1000
    labels = zeros(Int, ntrials)
    distances = [0.6348, 0.6589, 0.6435, 0.685, 0.7143]
    for i in 1:ntrials
        labels[i] = knn_classification(distances, classes_int; k=2)
    end
    frequencies = countmap(labels)
    @test 0.45 <= frequencies[2]/ntrials <= 0.55
    @test 0.45 <= frequencies[3]/ntrials <= 0.55
end

@testset "kNN classification - k=5 decrement" begin 
    distances = [0.6348, 0.6589, 0.6435, 0.685, 0.7143]
    label_int =  knn_classification(distances, classes_int; k=5, tie_break=:decrement)
    @test label_int == 3
    label_sym =  knn_classification(distances, classes_sym; k=5, tie_break=:decrement)
    @test label_sym == :science
end

@testset "kNN classification - k=5 min total" begin 
    distances = [0.6348, 0.6589, 0.6435, 0.685, 0.7143]
    label_int =  knn_classification(distances, classes_int; k=5, tie_break=:min_total)
    @test label_int == 3
    label_sym =  knn_classification(distances, classes_sym; k=5, tie_break=:min_total)
    @test label_sym == :science
end

@testset "kNN classification - decrement vs total" begin 
    distances = [0.4, 0.65, 0.5, 0.6, 0.1]
    labels = [1, 1, 2, 2, 3]
    label_dec = knn_classification(distances, labels; k=5, tie_break=:decrement)
    @test label_dec == 2
    # The tie break only works on the most common labels else class 3 would win.
    label_total = knn_classification(distances, labels; k=5, tie_break=:min_total)
    @test label_total == 1

    distances = [0.1, 0.4, 0.2, 0.3, 0.1]
    labels = [1, 1, 2, 2, 3]
    label_dec = knn_classification(distances, labels; k=5, tie_break=:decrement)
    @test label_dec == 2
    # For the next test, both 1 and 2 have a total of 0.5. 
    # This situation is considered rare for floating point numbers so it is not handled further.
    # Based on the dictionary construction 2 will always be chosen.
    label_total = knn_classification(distances, labels; k=5, tie_break=:min_total)
    @test label_total == 2
end

@testset "kNN classification multi - k=2 tie" begin 
    distances = [0.6348, 0.6589, 0.6435, 0.685, 0.7143]
    labels = knn_classification_multi(distances, classes_int; k=2)
    @test labels == [2, 3]
end
