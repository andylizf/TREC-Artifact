# test_conv2d_trec.py

import unittest

import torch
import torch.nn as nn

from trec.conv_layer import Conv2d_TREC

# |a - b| < atol + rtol * |b|
atol = 1e-6
rtol = 0.01


class TestConv2dTREC(unittest.TestCase):
    def setUp(self):
        self.input_tensor = torch.randn(
            1, 3, 64, 64, requires_grad=True).cuda()
        self.conv_standard = nn.Conv2d(3, 16, 3, stride=1, padding=1).cuda()
        self.conv_trec = Conv2d_TREC(
            3, 16, 3, param_L=9, param_H=8, layer=1, padding=1, stride=1).cuda()

        # Copy weights for fair comparison
        self.conv_trec.weight.data = self.conv_standard.weight.data.clone()
        if self.conv_standard.bias is not None:
            assert self.conv_trec.bias is not None
            self.conv_trec.bias.data = self.conv_standard.bias.data.clone()

    def test_forward(self):
        # Forward pass
        result_standard = self.conv_standard(self.input_tensor)
        result_trec = self.conv_trec(self.input_tensor)

        # Check if the outputs are almost equal
        self.assertTrue(torch.allclose(
            result_standard, result_trec, atol=atol, rtol=rtol))

    def test_backward(self):
        # Create a clone of the input tensor for independent manipulation in forward passes
        input_standard = self.input_tensor.clone().detach().requires_grad_(True)
        input_trec = self.input_tensor.clone().detach().requires_grad_(True)

        # Forward pass for standard convolution using a cloned input
        result_standard = self.conv_standard(input_standard)
        # Forward pass for custom TREC convolution using another cloned input
        result_trec = self.conv_trec(input_trec)

        # Create gradients for each backward pass
        grad_output_standard = torch.randn_like(result_standard).cuda()
        grad_output_trec = torch.randn_like(result_trec).cuda()

        # Backward pass for standard convolution
        result_standard.backward(grad_output_standard)
        # Backward pass for TREC convolution
        result_trec.backward(grad_output_trec)

        # Check if the gradients of the input tensors are almost equal
        assert input_standard.grad is not None and input_trec.grad is not None
        self.assertTrue(torch.allclose(
            input_standard.grad, input_trec.grad, atol=atol, rtol=rtol))

        # Check if the gradients of the weights are almost equal
        assert self.conv_standard.weight.grad is not None and self.conv_trec.weight.grad is not None
        self.assertTrue(torch.allclose(self.conv_standard.weight.grad,
                        self.conv_trec.weight.grad, atol=atol, rtol=rtol))

        # Optionally, check bias gradients if they exist
        if self.conv_standard.bias is not None and self.conv_trec.bias is not None:
            assert self.conv_standard.bias.grad is not None and self.conv_trec.bias.grad is not None
            self.assertTrue(torch.allclose(
                self.conv_standard.bias.grad, self.conv_trec.bias.grad, atol=atol, rtol=rtol))


if __name__ == "__main__":
    unittest.main()
