# custom_feature.py

"""
This module contains custom features for neural networks.
"""

import numpy as np
import tensorflow as tf

def CustomFeature_IDENT_X(x, _):
    """
    Custom feature: identity function for x.
    
    Args:
        x: Input tensor.
        _: Ignored input.
    
    Returns:
        The input tensor x.
    """
    return x

def CustomFeature_AND(x, y):
    """
    Custom feature: AND.
    
    Args:
        x: Input tensor.
        y: Input tensor.
    
    Returns:
        The logical AND of x and y.
    """
    return x & y

def CustomFeature_OR (x, y):
    """
    Custom feature: OR.
    
    Args:
        x: Input tensor.
        y: Input tensor.
    
    Returns:
        The logical OR of x and y.
    """
    return x | y

def CustomFeature_XOR(x, y):
    """
    Custom feature: XOR, exactly xor :D.
    
    Args:
        x: Input tensor.
        y: Input tensor.
    
    Returns:
        The logical XOR of x and y.
    """
    return x ^ y

def CustomFeature_X_AND_NOT_Y(x, y):
    """
    Custom feature: x AND NOT y.
    
    Args:
        x: Input tensor.
        y: Input tensor.
    
    Returns:
        The logical AND of x and the negation of y.
    """
    return x & ~y

__global_Rnd_00 = 0.0
__global_Rnd_01 = 0.0
__global_Rnd_10 = 0.0
__global_Rnd_11 = 1.1

def set_global_Rnd_Seed(value):
    """
    Set the global random seed and initialize global random variables.
    
    Args:
        value: Random seed value.
    """
    global __global_Rnd_00, __global_Rnd_01, __global_Rnd_10, __global_Rnd_11
    np.random.seed(value)
    __global_Rnd_00 = (np.random.randint(200) - 100) / 100
    __global_Rnd_01 = (np.random.randint(200) - 100) / 100
    __global_Rnd_10 = (np.random.randint(200) - 100) / 100
    __global_Rnd_11 = (np.random.randint(200) - 100) / 100  

def CustomFeature_RND(x, y):
    """
    Custom feature: random value based on input.
    
    Args:
        x: Input tensor.
        y: Input tensor.
    
    Returns:
        A random value associated to the inputs.
    """
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
    """
    Custom feature: slightly bent value based on input.
    
    Args:
        x: Input tensor.
        y: Input tensor.
    
    Returns:
        A slightly bent plane consisting of a set of 4 points that nearly form a plane, 
        creating a slight deviation to test the model's ability to find a separating boundary.
    """
    ret = (
        tf.cast((1 - x) * (1 - y), tf.float32) * __SLIGHTLY_BENT_00 + 
        tf.cast((1 - x) * y, tf.float32) * __SLIGHTLY_BENT_01 + 
        tf.cast(x * (1 - y), tf.float32) * __SLIGHTLY_BENT_10 + 
        tf.cast(x * y, tf.float32) * __SLIGHTLY_BENT_11
        )
    return ret


def __test_SlightlyBent():
    """
    Test the CustomFeature_SlightlyBent function.
    """
    # Test case 1: x and y are tensors
    x = tf.constant([0, 0, 1, 1])
    y = tf.constant([0, 1, 0, 1])
    LST_SLIGHTLY_BENT = [__SLIGHTLY_BENT_00, __SLIGHTLY_BENT_01, __SLIGHTLY_BENT_10, __SLIGHTLY_BENT_11]
    expected_output = tf.constant(LST_SLIGHTLY_BENT, dtype=tf.float32)
    assert tf.reduce_all(tf.equal(CustomFeature_SlightlyBent(x, y), expected_output))

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