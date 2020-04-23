import torch

a = torch.tensor([0.1, 0.2, 0.3, 0.4])
b1 = torch.tensor([True, True, False, True])
c1 = torch.tensor([0.6, 0.6, 0.6])

b2 = torch.tensor([False, False, True, False])
c2 = torch.tensor([0.6])

original_set = torch.Tensor.__setitem__.__get__
original_setitem = torch.Tensor.__setitem__
rewrap = False

import storch


# class wrap:
#     def __init__(self):
#         # self.level = 0
#         self.is_arg_none = True
#         self.arg0 = None
#
#     def __get__(self, *args):
#         # print("getting set")
#         # print(args)
#         self.is_arg_none = args[0] is None
#         self.arg0 = args[0]
#
#         return lambda *args: self.__call__(*args)
#
#     def __call__(self, *args):
#         storch.tensor.level += 1
#         # print("calling set", storch.tensor.level)
#         # print(args)
#         try:
#             if storch.tensor.level == 1 and not self.is_arg_none:
#                 # print("In here")
#                 original_setitem(self.arg0, *args)
#             else:
#                 original_setitem(*args)
#         except TypeError as e:
#             print(storch.tensor.level, e)
#             if storch.tensor.level == 1:
#                 original_setitem(self.arg0, *args)
#         finally:
#             storch.tensor.level -= 1


assert a[b2] == torch.tensor(0.3)
assert (a[b1] == torch.tensor([0.1, 0.2, 0.4])).all()

a = torch.tensor([0.1, 0.2, 0.3, 0.4])
a[b2] = c2
assert (a == torch.tensor([0.1, 0.2, 0.6, 0.4])).all()

a = torch.tensor([0.1, 0.2, 0.3, 0.4])
torch.Tensor.__setitem__(a, b2, c2)
# a[b2] = c2
assert (a == torch.tensor([0.1, 0.2, 0.6, 0.4])).all()

a = torch.tensor([0.1, 0.2, 0.3, 0.4])
a[b1] = c1
assert (a == torch.tensor([0.6, 0.6, 0.3, 0.6])).all()

a = torch.tensor([0.1, 0.2, 0.3, 0.4])
a[1] = 0.6
assert (a == torch.tensor([0.1, 0.6, 0.3, 0.4])).all()

a = torch.tensor([0.1, 0.2, 0.3, 0.4])
a[1:] = 0.6
assert (a == torch.tensor([0.1, 0.6, 0.6, 0.6])).all()

a = torch.tensor([0.1, 0.2, 0.3, 0.4])
a[a > 0.2] = 0.6
assert (a == torch.tensor([0.1, 0.2, 0.6, 0.6])).all()

a = torch.tensor([0.1, 0.2, 0.3, 0.4])
a[torch.tensor([2, 3])] = 0.6
assert (a == torch.tensor([0.1, 0.2, 0.6, 0.6])).all()

a = torch.tensor([0.1, 0.2, 0.3, 0.4])
a[1:] = torch.tensor([0.6, 0.7, 0.8])
assert (a == torch.tensor([0.1, 0.6, 0.7, 0.8])).all()

a = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
assert a[1, 2] == torch.tensor(0.6)
a[1, 2] = 0.7
assert (a == torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.7]])).all()

a = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
b = torch.tensor([[True, False, True], [False, True, False]])
assert (a[b] == torch.tensor([0.1, 0.3, 0.5])).all()
a[b] = 0.7
assert (a == torch.tensor([[0.7, 0.2, 0.7], [0.4, 0.7, 0.6]])).all()
# print(a[:])
# print("-------------------")
# print(a[b])
# print(torch.Tensor.__getitem__(a, b))
