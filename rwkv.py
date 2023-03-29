#! -*- coding: utf-8 -*-
# RWKV
# tensorflow 1.15 + bert4keras 0.11.4 测试通过

from bert4keras.layers import *


class RWKV(Layer):
    """RWKV
    链接1：https://github.com/BlinkDL/RWKV-LM
    链接2：https://kexue.fm/archives/9554
    """
    def __init__(
        self,
        units,
        use_bias=True,
        unroll=True,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        super(RWKV, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.unroll = unroll
        self.kernel_initializer = initializers.get(kernel_initializer)

    @integerize_shape
    def build(self, input_shape):
        super(RWKV, self).build(input_shape)
        hidden_size = input_shape[-1]
        self.rkv_dense = Dense(
            units=self.units * 3,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.o_dense = Dense(
            units=hidden_size,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

        def initializer(shape, dtype=None):
            r_min, r_max = 0.9, 0.999
            u = np.random.random(size=shape)
            return np.log(-0.5 * np.log(u * (r_max**2 - r_min**2) + r_min**2))

        self.nu_log = self.add_weight(
            name='nu_log', shape=(self.units,), initializer=initializer
        )
        self.gamma_log = self.add_weight(
            name='gamma_log', shape=(self.units,), initializer='zeros'
        )

    @recompute_grad
    def call(self, inputs, mask=None):
        rkv = self.rkv_dense(inputs)
        r, k, v = tf.split(rkv, 3, axis=-1)
        r, k = K.sigmoid(r), K.exp(k)
        kv = k * v
        u = K.concatenate([kv, k], axis=-1)
        nu = K.exp(K.concatenate([self.nu_log, self.nu_log], axis=0))
        gamma = K.exp(self.nu_log + self.gamma_log) - 1

        if self.unroll:
            L_in = K.int_shape(u)[1]
            assert L_in is not None, 'input_length can not be None while unroll=True'
            log2_L = int(np.ceil(np.log2(L_in)))
        else:
            L_in = K.shape(u)[1]
            log2_L = K.log(K.cast(L_in, K.floatx())) / K.log(2.)
            log2_L = K.cast(tf.ceil(log2_L), 'int32')

        u = tf.pad(u, [[0, 0], [0, 2**log2_L - K.shape(u)[1]], [0, 0]])
        B, L, D = K.shape(u)[0], K.shape(u)[1], K.int_shape(u)[-1]

        def rwkv(i, x):
            l = 2**i
            x = K.reshape(x, [B * L // l, l, D])
            x1, x2 = x[:, :l // 2], x[:, l // 2:]

            pos = K.arange(1, l // 2 + 1, dtype=K.floatx())
            nus = tf.einsum('n,d->nd', pos, nu)
            lambs = K.exp(-nus)

            x2 = x2 + lambs * x1[:, -1:]
            x = K.concatenate([x1, x2], axis=1)
            if (not self.unroll) and K.int_shape(u)[1] is not None:
                x = K.reshape(x, [B, L, D])

            return i + 1, x

        if self.unroll:
            for i in range(log2_L):
                _, u = rwkv(i + 1, u)
        else:
            _, u = tf.while_loop(lambda i, x: i <= log2_L, rwkv, [1, u])

        u1, u2 = tf.split(u[:, :L_in], 2, axis=-1)
        u = tf.math.divide_no_nan(u1 + gamma * kv, u2 + gamma * k) * r
        return self.o_dense(u)

    def get_config(self):
        config = {
            'units': self.units,
            'use_bias': self.use_bias,
            'unroll': self.unroll,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(RWKV, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
