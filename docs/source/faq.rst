FAQ
===

A method I'm using doesn't play well with the required independent dimensions in Storchastic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
An example of this is :func:`torch.nn.Conv2d`, which expects exactly an input of (N, C, H, W) and cannot have any more independent dimensions to the left of N. However, when sampling using Storchastic, we use the dimensions on the left to keep track of independent samples from different proposal distributions, meaning we might have an input of size (Z, N, C, H, W), which will not fit :func:`torch.nn.Conv2d`. Can we fix this? Yes!

The function :func:`storch.wrappers.make_left_broadcastable` helps us out there. It makes sure to flatten all independent dimensions into a single dimension before calling the function, and after calling the function it will restore them. You can call it using `make_left_broadcastable(Conv2d(16, 33, 3))`.