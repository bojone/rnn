#! -*- coding: utf-8 -*-
# RNN-α 模型实现
# tensorflow 1.15 + bert4keras 0.11.4 测试通过

from bert4keras.models import *
from lru import LRU
from slru import SLRU
from rwkv import RWKV

RNN = LRU  # SLRU、RWKV


class RNN_alpha(RoFormerV2):
    """RNN-α
    改动：基本模块换成RNN
    """
    def initializer(self, shape, dtype=None, order=2, gain=1.0):
        return super(RNN_alpha, self).initializer(shape, dtype, order, gain)

    def apply_main_layers(self, inputs, index):
        """RNN-α 的主体是基于RNN的模块
        顺序：RNN --> Add --> LN --> FFN --> Add --> LN
        """
        x = inputs
        rnn_name = 'Transformer-%d-RNN' % index
        ffn_name = 'Transformer-%d-FFN' % index

        xi = x
        x = self.apply(
            inputs=x,
            layer=RNN,
            units=(2 if RNN is SLRU else 1) * self.hidden_size,
            use_bias=False,
            kernel_initializer=self.initializer,
            name=rnn_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % rnn_name
        )
        x = self.apply(inputs=[xi, x], layer=Add, name='%s-Add' % rnn_name)
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=False,
            offset=False,
            epsilon=1e-12,
            name='%s-Norm' % rnn_name
        )

        xi = x
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            kernel_initializer=self.initializer,
            use_bias=False,
            name=ffn_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % ffn_name
        )
        x = self.apply(inputs=[xi, x], layer=Add, name='%s-Add' % rnn_name)
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=False,
            offset=False,
            epsilon=1e-12,
            name='%s-Norm' % ffn_name
        )

        return x
