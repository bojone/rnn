#! -*- coding: utf-8 -*-
# 简化版线性循环单元（Simpler Linear Recurrent Unit）
# tensorflow 1.15 + bert4keras 0.11.4 测试通过

from bert4keras.layers import *


class SLRU(Layer):
    """实数版线性循环单元
    链接1：https://arxiv.org/abs/2303.06349
    链接2：https://kexue.fm/archives/9554
    """
    def __init__(
        self,
        units,
        activation='linear',
        use_bias=True,
        unroll=True,  # unroll可以加速训练，但是会增加显存消耗
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        super(SLRU, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.unroll = unroll
        self.kernel_initializer = initializers.get(kernel_initializer)

    @integerize_shape
    def build(self, input_shape):
        super(SLRU, self).build(input_shape)
        hidden_size = input_shape[-1]
        self.i_dense = Dense(
            units=self.units,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.o_dense = Dense(
            units=hidden_size,
            use_bias=self.use_bias,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer
        )

        def initializer(shape, dtype=None):
            r_min, r_max = 0.9, 0.999
            u = np.random.random(size=shape[1])
            nu_log = np.log(-0.5 * np.log(u * (r_max**2 - r_min**2) + r_min**2))
            gamma_log = np.log(np.sqrt(1 - np.exp(-np.exp(nu_log))**2))
            return np.array([nu_log, gamma_log])

        self.params_log = self.add_weight(
            name='params_log', shape=(2, self.units), initializer=initializer
        )

    @recompute_grad
    def call(self, inputs, mask=None):
        u = self.i_dense(inputs)
        params = K.exp(self.params_log)
        nu, gamma = params[0], params[1]

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

        def lru(i, x):
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
            x = u
            for i in range(log2_L):
                _, x = lru(i + 1, x)
        else:
            _, x = tf.while_loop(lambda i, x: i <= log2_L, lru, [1, u])

        x = x[:, :L_in] * gamma
        return self.o_dense(x)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'unroll': self.unroll,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(SLRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
