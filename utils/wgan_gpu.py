# utils/wgan.py
import tensorflow as tf


@tf.function
def compute_gradient_penalty(critic, real_samples, fake_samples):
    """Gradient penalty for WGAN-GP (TensorFlow)."""
    batch_size = tf.shape(real_samples)[0]
    epsilon = tf.random.uniform(shape=(batch_size, 1), dtype=real_samples.dtype)
    interpolated = epsilon * real_samples + (1.0 - epsilon) * fake_samples

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        interpolated_scores = critic(interpolated, training=True)

    gradients = tape.gradient(interpolated_scores, interpolated)
    gradients = tf.reshape(gradients, [tf.shape(gradients)[0], -1])
    gradient_norm = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(gradients), axis=1))
    gradient_penalty = tf.reduce_mean(tf.square(1.0 - gradient_norm))
    return gradient_penalty
