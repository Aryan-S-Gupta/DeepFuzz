import pytest
import tensorflow as tf


def test_tensorflow_broadcast_to_base_valid():
    x = tf.constant([1.0, 1.0], dtype=tf.float32)
    y = tf.broadcast_to(x, [2, 2])
    assert tuple(y.shape) == (2, 2)
    assert y.dtype == tf.float32


def test_tensorflow_broadcast_to_shape_scalar_expected_negative():
    x = tf.constant([1.0, 1.0], dtype=tf.float32)
    with pytest.raises((TypeError, ValueError, tf.errors.InvalidArgumentError)):
        tf.broadcast_to(x, 1)
