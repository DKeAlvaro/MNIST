from mnist_optimized import test_model

import matplotlib.pyplot as plt

def test_model_with_different_num_neighbors():
    num_neighbors_list = [1, 3, 5, 10, 20, 30, 40, 50]
    accuracies = []
    
    for num_neighbors in num_neighbors_list:
        accuracy = test_model(num_neighbors)
        accuracies.append(accuracy)
    
    plt.plot(num_neighbors_list, accuracies, marker='o')
    plt.title('Model Accuracy vs Number of Neighbors')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.xticks(num_neighbors_list)
    plt.grid()
    plt.show()

if __name__ == "__main__":
    test_model_with_different_num_neighbors()

