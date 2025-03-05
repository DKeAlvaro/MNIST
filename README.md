<p style="text-align: center;">
    <img src="banner.png" alt="Banner" style="max-width: 80%; height: auto;">
</p>

# MNIST Digit Recognition

Achieved 75% accuracy using a simple cosine similarity approach under 50 lines of code, with no training.
Each instance from the test set is compared with `num_neighbors` instances of each digit from the training set, and the digit with the highest sum of similarities is predicted. As we can see in the plot below, the accuracy increases with the number of neighbors.

<p style="text-align: center;">
    <img src="results.png" alt="Results" style="max-width: 40%; height: auto;">
</p>
