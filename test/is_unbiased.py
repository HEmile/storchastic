import storch
import torch
from torch.distributions import Bernoulli, OneHotCategorical

expect = storch.method.Expect("x")
probs = torch.tensor([0.95, 0.01, 0.01, 0.01, 0.01, 0.01], requires_grad=True)
indices = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
b = OneHotCategorical(probs=probs)
z = expect.sample(b)
c = (2.4 * z * indices).sum(-1)
storch.add_cost(c, "no_baseline_cost")

storch.backward()

expect_grad = z.grad["probs"].clone()


def eval(grads):
    print("----------------------------------")
    grad_samples = storch.gather_samples(grads, "variance")
    mean = storch.reduce_plates(grad_samples, plates=["variance"])
    print("mean grad", mean)
    print("expected grad", expect_grad)
    print("specific_diffs", (mean - expect_grad) ** 2)
    mse = storch.reduce_plates((grad_samples - expect_grad) ** 2).sum()
    print("MSE", mse)
    bias = (storch.reduce_plates((mean - expect_grad) ** 2)).sum()
    print("bias", bias)
    return bias


# method = storch.method.UnorderedSetEstimator("x", k=6)
# method = storch.REBAR()
# method = storch.method.ScoreFunction(
#     "x", baseline_factory="batch_average", n_samples=10
# )
# method = storch.method.ScoreFunction("x", baseline_factory="moving_average")
method = storch.method.RELAX("x", in_dim=6)
# method = storch.method.RELAX("x", in_dim=6)
optim = torch.optim.SGD(method.parameters(), lr=0.000001)
tot_bias = 0
for _ in range(100):
    grads = []
    print(method.eta, method.temperature)
    for i in range(100):
        optim.zero_grad()
        b = OneHotCategorical(probs=probs)
        z = method.sample(b)
        c = (2.4 * z * indices).sum(-1) + 100
        storch.add_cost(c, "baseline_cost")

        storch.backward(update_estimator_params=True)
        grad = z.grad["probs"].clone()
        grads.append(grad)
        if (i + 1) % 1000 == 0:
            print(i)
            eval(grads)
        optim.step()

    # print("FINAL EVAL")
    tot_bias += eval(grads)

print(tot_bias / 100)
# expected_estim_grad = 0.0
# for i in range(6):
#     to_predict = [0] * 6
#     to_predict[i] = 1
#     to_predict = torch.tensor(to_predict)
#     b = OneHotCategorical(probs=probs)
#     while True:
#         z = method.sample(b)
#         _z = z._tensor
#         if _z.ndim == 2:
#             _z = _z[0]
#         if torch.argmax(_z) == i:
#             break
#     c = (2.4 * z * indices).sum(-1) + 100
#     storch.add_cost(c, "baseline_cost")
#
#     storch.backward(update_estimator_params=False)
#     grad = z.grad["probs"].clone()
#     expected_estim_grad += grad * probs[i]
# print("estimated grad", expected_estim_grad)
# print("expected grad", expect_grad)
# bias = ((expected_estim_grad - expect_grad) ** 2).sum()
# print("bias", bias)
