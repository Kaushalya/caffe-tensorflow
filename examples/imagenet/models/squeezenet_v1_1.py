from kaffe.tensorflow import Network


class Squeezenet(Network):
    def setup(self):
        (self.feed('data')
             .conv(3, 3, 64, 2, 2, padding='VALID', name='conv1')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .conv(1, 1, 16, 1, 1, name='fire2_squeeze1x1')
             .conv(1, 1, 64, 1, 1, name='fire2_expand1x1'))

        (self.feed('fire2_squeeze1x1')
             .conv(3, 3, 64, 1, 1, name='fire2_expand3x3'))

        (self.feed('fire2_expand1x1',
                   'fire2_expand3x3')
             .concat(3, name='fire2_concat')
             .conv(1, 1, 16, 1, 1, name='fire3_squeeze1x1')
             .conv(1, 1, 64, 1, 1, name='fire3_expand1x1'))

        (self.feed('fire3_squeeze1x1')
             .conv(3, 3, 64, 1, 1, name='fire3_expand3x3'))

        (self.feed('fire3_expand1x1',
                   'fire3_expand3x3')
             .concat(3, name='fire3_concat')
             .max_pool(3, 3, 2, 2, name='pool3')
             .conv(1, 1, 32, 1, 1, name='fire4_squeeze1x1')
             .conv(1, 1, 128, 1, 1, name='fire4_expand1x1'))

        (self.feed('fire4_squeeze1x1')
             .conv(3, 3, 128, 1, 1, name='fire4_expand3x3'))

        (self.feed('fire4_expand1x1',
                   'fire4_expand3x3')
             .concat(3, name='fire4_concat')
             .conv(1, 1, 32, 1, 1, name='fire5_squeeze1x1')
             .conv(1, 1, 128, 1, 1, name='fire5_expand1x1'))

        (self.feed('fire5_squeeze1x1')
             .conv(3, 3, 128, 1, 1, name='fire5_expand3x3'))

        (self.feed('fire5_expand1x1',
                   'fire5_expand3x3')
             .concat(3, name='fire5_concat')
             .max_pool(3, 3, 2, 2, name='pool5')
             .conv(1, 1, 48, 1, 1, name='fire6_squeeze1x1')
             .conv(1, 1, 192, 1, 1, name='fire6_expand1x1'))

        (self.feed('fire6_squeeze1x1')
             .conv(3, 3, 192, 1, 1, name='fire6_expand3x3'))

        (self.feed('fire6_expand1x1',
                   'fire6_expand3x3')
             .concat(3, name='fire6_concat')
             .conv(1, 1, 48, 1, 1, name='fire7_squeeze1x1')
             .conv(1, 1, 192, 1, 1, name='fire7_expand1x1'))

        (self.feed('fire7_squeeze1x1')
             .conv(3, 3, 192, 1, 1, name='fire7_expand3x3'))

        (self.feed('fire7_expand1x1',
                   'fire7_expand3x3')
             .concat(3, name='fire7_concat')
             .conv(1, 1, 64, 1, 1, name='fire8_squeeze1x1')
             .conv(1, 1, 256, 1, 1, name='fire8_expand1x1'))

        (self.feed('fire8_squeeze1x1')
             .conv(3, 3, 256, 1, 1, name='fire8_expand3x3'))

        (self.feed('fire8_expand1x1',
                   'fire8_expand3x3')
             .concat(3, name='fire8_concat')
             .conv(1, 1, 64, 1, 1, name='fire9_squeeze1x1')
             .conv(1, 1, 256, 1, 1, name='fire9_expand1x1'))

        (self.feed('fire9_squeeze1x1')
             .conv(3, 3, 256, 1, 1, name='fire9_expand3x3'))

        (self.feed('fire9_expand1x1',
                   'fire9_expand3x3')
             .concat(3, name='fire9_concat')
             .conv(1, 1, 1000, 1, 1, name='conv10')
             .avg_pool(14, 14, 1, 1, padding='VALID', name='pool10')
         .softmax(name='prob'))