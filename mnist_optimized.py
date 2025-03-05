import tensorflow as tf
import numpy as np
from scipy.special import softmax

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Subsets for each number
number_subsets = [x_train[y_train == num] for num in range(10)]

def get_examples(number, num_samples):
    """Get multiple random examples of a number"""
    subset = number_subsets[number]
    indices = np.random.randint(0, len(subset), num_samples)
    return subset[indices]

def batch_cosine_similarity(a, b_batch):
    a_flat = a.flatten()
    b_flat = b_batch.reshape(len(b_batch), -1)
    dot_products = np.dot(b_flat, a_flat)
    magnitude_a = np.linalg.norm(a_flat)
    magnitude_b = np.linalg.norm(b_flat, axis=1)
    cosine_similarities = dot_products / (magnitude_a * magnitude_b)
    return cosine_similarities


def predict_number(x_instance, num_neighbors=10):
    distances = np.zeros(10)
    for number in range(10):
        examples = get_examples(number, num_neighbors)
        similarities = batch_cosine_similarity(x_instance, examples)
        distances[number] = np.sum(similarities)
    return np.argmax(softmax(distances))

def test_model(num_neighbors=10):
    correct = wrong = 0
    for i, (img, true_label) in enumerate(zip(x_test, y_test)):
        pred = predict_number(img, num_neighbors)
        correct += (pred == true_label)
        wrong += (pred != true_label)
        if (i+1) % 100 == 0: 
            print(f"Tested {i+1}: Accuracy {correct/(correct+wrong):.4f}")
    print(f"Final Accuracy: {correct/(correct+wrong):.4f}")
    return correct/(correct+wrong)

if __name__ == "__main__":  
    test_model(num_neighbors=10)