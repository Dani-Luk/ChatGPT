# custom_feature.py
import numpy as np
import tensorflow as tf

def CustomFeature_IDENT_X(x, _):
    """Custom feature: identity function for x"""
    return x

def CustomFeature_AND(x, y):
    """Custom feature: AND"""
    return x & y

def CustomFeature_OR (x, y):
    """Custom feature: OR"""
    return x & y

def CustomFeature_XOR(x, y):
    """Custom feature: XOR , exactly xor :D """
    return x ^ y

def CustomFeature_X_AND_NOT_Y(x, y):
    """Custom feature: x AND NOT y"""
    return x & ~y

__global_Rnd_00 = 0.0
__global_Rnd_01 = 0.0
__global_Rnd_10 = 0.0
__global_Rnd_11 = 1.1

def set_global_Rnd_Seed(value):
    global __global_Rnd_00, __global_Rnd_01, __global_Rnd_10, __global_Rnd_11
    np.random.seed(value)
    __global_Rnd_00 = (np.random.randint(200) - 100) / 100
    __global_Rnd_01 = (np.random.randint(200) - 100) / 100
    __global_Rnd_10 = (np.random.randint(200) - 100) / 100
    __global_Rnd_11 = (np.random.randint(200) - 100) / 100  

def CustomFeature_RND(x, y):
    ret = (
        tf.cast((1 - x) * (1 - y), tf.float32) * __global_Rnd_00 + 
        tf.cast((1 - x) * y, tf.float32) * __global_Rnd_01 + 
        tf.cast(x * (1 - y), tf.float32) * __global_Rnd_10 + 
        tf.cast(x * y, tf.float32) * __global_Rnd_11
        )
    return ret

__SLIGHTLY_BENT_00 = 0.0
__SLIGHTLY_BENT_01 = 1.0
__SLIGHTLY_BENT_10 = 1.0
__SLIGHTLY_BENT_11 = 1.95

def CustomFeature_SlightlyBent(x, y):
    ret = (
        tf.cast((1 - x) * (1 - y), tf.float32) * __SLIGHTLY_BENT_00 + 
        tf.cast((1 - x) * y, tf.float32) * __SLIGHTLY_BENT_01 + 
        tf.cast(x * (1 - y), tf.float32) * __SLIGHTLY_BENT_10 + 
        tf.cast(x * y, tf.float32) * __SLIGHTLY_BENT_11
        )
    return ret


def __test_SlightlyBent():
    # Test case 1: x and y are tensors
    x = tf.constant([0, 0, 1, 1])
    y = tf.constant([0, 1, 0, 1])
    expected_output = tf.constant([0, 1, 1, 1.9], dtype=tf.float32)
    # assert tf.reduce_all(tf.equal(SlightlyBent(x, y), expected_output))

    # Test case 2: x and y are numpy arrays
    x = np.array([0, 0, 1, 1], dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.float32)
    expected_output = np.array([__SLIGHTLY_BENT_00, __SLIGHTLY_BENT_01, __SLIGHTLY_BENT_10, __SLIGHTLY_BENT_11], dtype=np.float32)
    assert np.array_equal(CustomFeature_SlightlyBent(x, y), expected_output)

    # Test case 3: x and y are integers
    assert CustomFeature_SlightlyBent(0, 0) == __SLIGHTLY_BENT_00
    assert CustomFeature_SlightlyBent(0, 1) == __SLIGHTLY_BENT_01
    assert CustomFeature_SlightlyBent(1, 0) == __SLIGHTLY_BENT_10
    assert CustomFeature_SlightlyBent(1, 1) == __SLIGHTLY_BENT_11

    print("All SlightlyBent test cases passed!")

if __name__ == "__main__":
    __test_SlightlyBent()