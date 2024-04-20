import torch.autograd

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = -ctx.lambda_val * grad_output
        return grad_input, None
