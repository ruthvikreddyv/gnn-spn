import os
import tensorflow as tf
import numpy as np
from gat_encoder import GATEncoder


class GNNSPNModel(tf.keras.Model):

    def __init__(
        self,
        n_node_features=82,
        n_actions=60,
        gat_layers=2,
        gat_hidden_dim=64,
        gat_out_dim=64,
        gat_heads=4,
        mlp_hidden=128,
        lr_sup=1e-3,
        lr_sup_decay=0.97,
        lr_sup_steps=2000,
        lr_minimum=1e-5,
        lr_ppo=3e-4,
        entropy_coef=0.01,
        vf_coef=0.5,
        clip_epsilon=0.2,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.clip_epsilon = clip_epsilon

        self.gat = GATEncoder(
            n_layers=gat_layers,
            hidden_dim=gat_hidden_dim,
            out_dim=gat_out_dim,
            n_heads=gat_heads,
            name="gat_encoder",
        )

        init = tf.keras.initializers.GlorotUniform()

        self.pol_W1 = tf.keras.layers.Dense(mlp_hidden, activation="relu", kernel_initializer=init, name="pol_W1")
        self.pol_W2 = tf.keras.layers.Dense(n_actions, activation=None, kernel_initializer=init, name="pol_W2")

        self.val_W1 = tf.keras.layers.Dense(mlp_hidden, activation="relu", kernel_initializer=init, name="val_W1")
        self.val_W2 = tf.keras.layers.Dense(1, activation=None, kernel_initializer=init, name="val_W2")

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr_sup,
            decay_steps=lr_sup_steps,
            decay_rate=lr_sup_decay,
            staircase=True,
        )

        class _MinLR:
            def __init__(self, sched, mn):
                self._s = sched
                self._m = mn
                self.iterations = tf.Variable(0, trainable=False, dtype=tf.int64)
            def __call__(self):
                return tf.maximum(self._s(self.iterations), self._m)
            def get_config(self):
                return {}

        self.sup_optimizer = tf.keras.optimizers.Adam(learning_rate=_MinLR(lr_schedule, lr_minimum), epsilon=1e-7)
        self.ppo_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_ppo, epsilon=1e-5, clipnorm=0.5)

    def call(self, inputs, training=False):
        X, A = inputs
        X = tf.cast(X, tf.float32)
        A = tf.cast(A, tf.float32)

        Z = self.gat((X, A), training=training)

        logits = self.pol_W2(self.pol_W1(Z))
        probs = tf.nn.softmax(logits, axis=-1)

        z_mean = tf.reduce_mean(Z, axis=0, keepdims=True)
        value = tf.squeeze(self.val_W2(self.val_W1(z_mean)))

        return logits, probs, value

    def supervised_step(self, X, A, labels):
        X = tf.cast(X, tf.float32)
        A = tf.cast(A, tf.float32)
        labels = tf.cast(labels, tf.float32)

        with tf.GradientTape() as tape:
            logits, probs, _ = self((X, A), training=True)
            logits_c = tf.clip_by_value(logits, -30.0, 30.0)
            log_probs = tf.nn.log_softmax(logits_c, axis=-1)
            loss = -tf.reduce_mean(tf.reduce_sum(labels * log_probs, axis=-1))

        if not tf.math.is_nan(loss):
            grads = tape.gradient(loss, self.trainable_variables)
            gv = [(g, v) for g, v in zip(grads, self.trainable_variables) if g is not None]
            g_list = [g for g, v in gv]
            v_list = [v for g, v in gv]
            g_list, _ = tf.clip_by_global_norm(g_list, 1.0)
            self.sup_optimizer.apply_gradients(zip(g_list, v_list))
        return loss, probs

    @tf.function
    def ppo_step(self, X, A, actions, old_log_probs, returns, advantages):
        X = tf.cast(X, tf.float32)
        A = tf.cast(A, tf.float32)
        actions = tf.cast(actions, tf.int32)
        old_log_probs = tf.cast(old_log_probs, tf.float32)
        returns = tf.cast(returns, tf.float32)
        advantages = tf.cast(advantages, tf.float32)

        with tf.GradientTape() as tape:
            logits, probs, value = self((X, A), training=True)

            log_probs_all = tf.nn.log_softmax(logits, axis=-1)
            action_mask = tf.one_hot(actions, self.n_actions)
            log_probs = tf.reduce_sum(log_probs_all * action_mask, axis=-1)

            ratio = tf.exp(log_probs - old_log_probs)
            clip_r = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            pol_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clip_r * advantages))

            val_loss = tf.reduce_mean(tf.square(returns - value))

            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1)
            ent_loss = -tf.reduce_mean(entropy)

            total = pol_loss + self.vf_coef * val_loss + self.entropy_coef * ent_loss

        grads = tape.gradient(total, self.trainable_variables)
        self.ppo_optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {
            "total": total,
            "policy": pol_loss,
            "value": val_loss,
            "entropy": tf.reduce_mean(entropy),
        }

    def get_action(self, X, A, deterministic=False):
        X = tf.cast(X, tf.float32)
        A = tf.cast(A, tf.float32)
        logits, probs, value = self((X, A), training=False)

        if deterministic:
            actions = tf.argmax(probs, axis=-1).numpy().astype(np.int32)
        else:
            actions = tf.squeeze(
                tf.random.categorical(tf.math.log(probs + 1e-8), 1), axis=-1
            ).numpy().astype(np.int32)

        log_probs_all = tf.nn.log_softmax(logits, axis=-1)
        action_mask = tf.one_hot(actions, self.n_actions)
        log_probs = tf.reduce_sum(
            log_probs_all * tf.cast(action_mask, tf.float32), axis=-1
        ).numpy()

        return actions, log_probs, float(value.numpy())

    def save_weights_to_file(self, path="weight/gnn_spn_best.h5"):
        if os.path.exists(path):
            os.remove(path)
        self.save_weights(path)
        print("[model] Saved -> " + path)

    def load_weights_from_file(self, path="weight/gnn_spn_best.h5"):
        self.load_weights(path)
        print("[model] Loaded <- " + path)
