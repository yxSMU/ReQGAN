import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
import tensorcircuit as tc

tc.set_backend("tensorflow")


class ClassicalCritic(Model):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = layers.Dense(512)
        self.fc2 = layers.Dense(256)
        self.fc3 = layers.Dense(1)

    def call(self, x, training=False):
        x = tf.nn.leaky_relu(self.fc1(x), 0.2)
        x = tf.nn.leaky_relu(self.fc2(x), 0.2)
        return self.fc3(x)


class AngleNet(Model):
    """
    Output:
      - angles: [B, Q] mapped to (-pi, pi)
      - aux1:  [B, 1] raw   (for a)
      - aux2:  [B, 1] raw   (for b)
      - aux3:  [B, 1] raw   (for u0, center of u)
    """

    def __init__(self, latent_dim, n_data_qubits, hidden=256):
        super().__init__()
        self.n_data_qubits = int(n_data_qubits)
        self.net = tf.keras.Sequential(
            [
                layers.Dense(hidden, activation="swish"),
                layers.Dense(hidden, activation="swish"),
                layers.Dense(self.n_data_qubits + 3),  # Q + aux1 + aux2 + aux3(u0)
            ]
        )

    def call(self, z, training=False):
        x = self.net(z, training=training)  # [B, Q+3]
        angles_raw = x[:, : self.n_data_qubits]  # [B,Q]
        aux_raw = x[:, self.n_data_qubits : self.n_data_qubits + 3]  # [B,3]

        angles = np.pi * tf.tanh(0.5 * angles_raw)  # (-pi, pi)
        aux1 = aux_raw[:, 0:1]  # [B,1]
        aux2 = aux_raw[:, 1:2]  # [B,1]
        aux3 = aux_raw[:, 2:3]  # [B,1]  (u0 raw)

        return angles, aux1, aux2, aux3


class QuantumGeneratorAncillaReuploadTC(Model):
    def __init__(
        self,
        n_data_qubits,
        n_total_qubits,
        n_layers,
        latent_dim,
        full_data_dim,
        output_dim,
    ):
        super().__init__()
        self.n_data_qubits = int(n_data_qubits)
        self.n_total_qubits = int(n_total_qubits)
        self.n_layers = int(n_layers)
        self.latent_dim = int(latent_dim)
        self.full_data_dim = int(full_data_dim)
        self.output_dim = int(output_dim)

        self.angle_net = AngleNet(latent_dim, self.n_data_qubits)

        self.tc_circuit_layer = tc.KerasLayer(
            self.quantum_circuit,
            weights_shape=(self.n_layers, self.n_data_qubits, 3),
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(0.0, 0.05),
        )

    def quantum_circuit(self, inp, weights):
        c = tc.Circuit(self.n_total_qubits)
        Q = self.n_data_qubits

        # inp: [Q] init_angles
        init_angles = tf.reshape(inp, (Q,))  # ensure [Q]

        for q in range(Q):
            c.ry(q, theta=init_angles[q])

        for l in range(self.n_layers):
            for q in range(Q):
                c.ry(q, theta=weights[l, q, 0])
                c.rz(q, theta=weights[l, q, 1])
                c.ry(q, theta=weights[l, q, 2])
            for q in range(self.n_total_qubits - 1):
                c.cnot(q, q + 1)

        return c.state()

    def call(self, z, training=False, return_parts=False):
        eps = tf.constant(1e-6, tf.float32)

        # --------- hyperparams ---------
        k = tf.constant(1, tf.float32)  # softplus(k*u) hardness, try 1~10
        temp_sb = tf.constant(1, tf.float32)  # head strength, try 0.5~2

        z = tf.cast(z, tf.float32)

        # --------- AngleNet outputs ---------
        init_angles, aux1, aux2, aux3 = self.angle_net(
            z, training=training
        )  # [B,Q], [B,1], [B,1], [B,1]

        # --------- quantum state ---------
        inp_main = init_angles
        state = self.tc_circuit_layer(
            inp_main, training=training
        )  # [B, 2^(Q+1)] complex

        amp0 = state[:, 0::2]
        amp1 = state[:, 1::2]

        p0 = tf.abs(amp0) ** 2
        p1 = tf.abs(amp1) ** 2

        p_anc0 = tf.reduce_sum(p0, axis=1, keepdims=True)
        p_anc1 = tf.reduce_sum(p1, axis=1, keepdims=True)

        p0_cond = p0 / (p_anc0 + eps)  # [B, N], sum=1
        p1_cond = p1 / (p_anc1 + eps)

        tau = 2.1
        logp = tf.math.log(tf.clip_by_value(p0_cond, 1e-12, 1.0))
        p_smooth = tf.nn.softmax(logp / tau, axis=1)

        # =========================================================
        # Pixel mapping with learned u-center
        # u = N p - 1, then center by u0: u_centered = u - u0
        # x_feat = softplus(k*u_centered) - log(2)
        # =========================================================
        N = tf.cast(self.full_data_dim, tf.float32)  # N = 2^Q (e.g., 1024)

        u = N * p_smooth - 1.0  # [B, N]

        x_feat = u

        # Layer normalization

        mu = tf.reduce_mean(x_feat, axis=1, keepdims=True)
        sigma = tf.math.reduce_std(x_feat, axis=1, keepdims=True)
        x_feat = (x_feat - mu) / (sigma + 1e-6)

        # --------- a,b heads ---------
        a = 2.0 * tf.sigmoid(temp_sb * aux1)  # (0,5)
        b = -tf.nn.softplus(temp_sb * aux2)  # <= 0
        v = a * x_feat + b

        x01 = tf.sigmoid(v)  # (0,1)

        x11 = 2.0 * x01 - 1.0  # (-1,1)
        m01 = tf.reduce_mean(x01, axis=1, keepdims=True)

        if return_parts:
            return x11, p0_cond, p_smooth, p_anc0, p_anc1, m01
        return x11


class PQWGAN_CC(Model):
    def __init__(self, image_size, channels, n_data_qubits, n_layers, latent_dim):
        super().__init__()
        assert image_size == 32 and channels == 1
        self.full_data_dim = 2**n_data_qubits
        self.output_dim = image_size * image_size

        self.critic = ClassicalCritic(self.output_dim)
        self.generator = QuantumGeneratorAncillaReuploadTC(
            n_data_qubits=n_data_qubits,
            n_total_qubits=n_data_qubits + 1,
            n_layers=n_layers,
            latent_dim=latent_dim,
            full_data_dim=self.full_data_dim,
            output_dim=self.output_dim,
        )
