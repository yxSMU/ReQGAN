# models/size16_pqwgan.py
import tensorflow as tf
from tensorflow.keras import Model, layers
import tensorcircuit as tc
import numpy as np

tc.set_backend("tensorflow")


class ClassicalCritic(Model):
    """ """

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = layers.Dense(512)
        self.fc2 = layers.Dense(256)
        self.fc3 = layers.Dense(1)

    def call(self, x, training=False):
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        x = tf.nn.leaky_relu(self.fc1(x), 0.2)
        x = tf.nn.leaky_relu(self.fc2(x), 0.2)
        return self.fc3(x)


class QuantumGeneratorNoPatchNoAngleNet(Model):
    """
    - latent_dim == n_qubits == n_data_qubits + 1 ancilla
    - RY(latent[i]) on all qubits
    """

    def __init__(self, n_qubits: int, n_ancillas: int, n_layers: int, output_dim: int):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.n_ancillas = int(n_ancillas)
        self.n_layers = int(n_layers)
        self.output_dim = int(output_dim)

        if self.n_ancillas != 1:
            raise ValueError("This implementation requires one ancilla qubit.")
        if self.n_qubits <= self.n_ancillas:
            raise ValueError("n_qubits must be > n_ancillas.")
        self.n_data_qubits = self.n_qubits - self.n_ancillas

        self.patch_size = 2**self.n_data_qubits
        if self.patch_size < self.output_dim:
            raise ValueError(
                f"2**(n_qubits-n_ancillas) must be >= output_dim. "
                f"Got {self.patch_size} < {self.output_dim}"
            )

        self.tc_layer = tc.KerasLayer(
            self._circuit_state,
            weights_shape=(self.n_layers, self.n_qubits, 3),
            dtype=tf.float32,
            initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0),
        )

    def _circuit_state(self, latent_vec_1d, weights):
        c = tc.Circuit(self.n_qubits)

        for q in range(self.n_qubits):
            c.ry(q, theta=latent_vec_1d[q])

        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                c.rz(q, theta=weights[l, q, 0])
                c.ry(q, theta=weights[l, q, 1])
                c.rz(q, theta=weights[l, q, 2])
            for q in range(self.n_qubits - 1):
                c.cnot(q, q + 1)

        return c.state()

    def call(self, z, training=False, ancilla_lowest_bit: bool = False):
        z = tf.cast(z, tf.float32)
        state = self.tc_layer(z, training=training)  # [B, 2**n_qubits] complex

        probs = tf.abs(state) ** 2

        half = 2**self.n_data_qubits

        if not ancilla_lowest_bit:
            probs_anc0 = probs[:, :half]
        else:
            probs_anc0 = probs[:, 0::2]

            # p0 = tf.reduce_sum(probs_anc0, axis=1)  # [B]
            # tf.print("P(a=0): mean=", tf.reduce_mean(p0),
            #         "std=", tf.math.reduce_std(p0),
            #         "min=", tf.reduce_min(p0),
            #         "max=", tf.reduce_max(p0))

            # if not ancilla_lowest_bit:
            #     probs_anc0_raw = probs[:, :half]      # [B, half]
            #     probs_anc1_raw = probs[:, half:]      # [B, half]
            # else:
            #     probs_anc0_raw = probs[:, 0::2]       # [B, half]
            #     probs_anc1_raw = probs[:, 1::2]       # [B, half]

            # p_x_raw = probs_anc0_raw + probs_anc1_raw

            # p_x = p_x_raw / (tf.reduce_sum(p_x_raw, axis=1, keepdims=True) + 1e-12)
            # p_x_a0 = probs_anc0_raw / (tf.reduce_sum(probs_anc0_raw, axis=1, keepdims=True) + 1e-12)

            # l1 = tf.reduce_mean(tf.reduce_sum(tf.abs(p_x - p_x_a0), axis=1))

            # p0 = tf.reduce_sum(probs_anc0_raw, axis=1)
            # tf.print("P(a=0) mean=", tf.reduce_mean(p0),
            #         "L1(p(x),p(x|a=0))=", l1)

            # # p0 = tf.reduce_sum(probs_anc0, axis=1)  # [B]  = P(ancilla=0)
            # # p1 = 1.0 - p0

            # # tf.print("P(a=0): mean=", tf.reduce_mean(p0),
            # #         "min=", tf.reduce_min(p0),
            # #         "max=", tf.reduce_max(p0),
            # #         "  P(a=1) mean=", tf.reduce_mean(p1))

        probs_anc0 = probs_anc0 / (
            tf.reduce_sum(probs_anc0, axis=1, keepdims=True) + 1e-12
        )

        maxv = tf.reduce_max(probs_anc0, axis=1, keepdims=True) + 1e-12
        post_processed = ((probs_anc0 / maxv) - 0.5) * 2.0  # [-1,1]

        return post_processed[:, : self.output_dim]


class PQWGAN_CC(Model):
    """ """

    def __init__(
        self,
        image_size: int = 32,
        channels: int = 1,
        n_data_qubits: int = 10,
        n_ancillas: int = 1,
        n_layers: int = 20,
    ):
        super().__init__()
        if channels != 1:
            raise ValueError("Only single-channel grayscale images are supported.")

        self.image_size = int(image_size)
        self.output_dim = self.image_size * self.image_size
        self.n_data_qubits = int(n_data_qubits)
        self.n_ancillas = int(n_ancillas)
        self.n_qubits = self.n_data_qubits + self.n_ancillas

        self.critic = ClassicalCritic(self.output_dim)
        self.generator = QuantumGeneratorNoPatchNoAngleNet(
            n_qubits=self.n_qubits,
            n_ancillas=self.n_ancillas,
            n_layers=n_layers,
            output_dim=self.output_dim,
        )
