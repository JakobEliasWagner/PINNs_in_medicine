import unittest

import numpy as np
import torch

from src.layers.scale_layer import ScaleLayer


class ScaleLayerTestCase(unittest.TestCase):
    def test_inner(self):
        n = 100
        tensor_size = 128
        for _ in range(n):
            # setup layer
            before_bv = np.random.random((2,))
            after_bv = np.random.random((2,))
            layer = ScaleLayer(
                min(before_bv), max(before_bv), min(after_bv), max(after_bv)
            )

            # pass tensor through layer
            x = (min(before_bv) - max(before_bv)) * torch.rand(
                tensor_size, tensor_size
            ) + max(before_bv)
            out = layer(x)

            left_cond = torch.all(torch.greater_equal(out, min(after_bv)))
            right_cond = torch.all(torch.less_equal(out, min(after_bv)))

            self.assertTrue(left_cond and right_cond)

    def test_left_boundary(self):
        n = 100
        for _ in range(n):
            # setup layer
            before_bv = np.random.random((2,))
            after_bv = np.random.random((2,))
            layer = ScaleLayer(
                min(before_bv), max(before_bv), min(after_bv), max(after_bv)
            )

            # pass tensor through layer
            x = torch.Tensor([min(before_bv)], dtype=torch.float32)
            out = layer(x)

            cond = torch.equal(out, min(after_bv))

            self.assertTrue(cond)

    def test_right_boundary(self):
        n = 100
        for _ in range(n):
            # setup layer
            before_bv = np.random.random((2,))
            after_bv = np.random.random((2,))
            layer = ScaleLayer(
                min(before_bv), max(before_bv), min(after_bv), max(after_bv)
            )

            # pass tensor through layer
            x = torch.Tensor([max(before_bv)], dtype=torch.float32)
            out = layer(x)

            cond = torch.equal(out, max(after_bv))

            self.assertTrue(cond)
