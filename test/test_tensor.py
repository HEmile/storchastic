from storch.tensor import Plate
import pytest
import torch
import storch

n = {1, 5, 1000, -1}


@pytest.fixture
def plate():
    return Plate("test", 123, [Plate("parent", 31, [])])


def test_plate(plate):
    assert False


def test_reduce(plate):
    assert False


def test_on_collecting_args(plate):
    assert False


@pytest.mark.parametrize("n", n)
def test_on_unwrap_tensor(plate, n):
    assert False


def to_storch(tensor: torch.Tensor) -> storch.Tensor:
    return storch.Tensor(tensor, [], [], "test")


@pytest.fixture
def a() -> torch.Tensor:
    return torch.tensor([0.1, 0.2, 0.3, 0.4])


@pytest.fixture
def b() -> torch.Tensor:
    return torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


@pytest.fixture
def mask(request):
    return torch.tensor(request.param)


@pytest.fixture
def expected(request):
    return torch.tensor(request.param)


@pytest.fixture
def value(request):
    return torch.tensor(request.param)


def c1() -> torch.Tensor:
    return torch.tensor([0.6, 0.6, 0.6])


def test_bool(a):
    a = to_storch(a)
    with pytest.raises(storch.IllegalStorchExposeError):
        if a[0] < 0.3:
            assert False


@pytest.mark.parametrize(
    "mask,expected",
    [
        ([False, False, True, False], 0.3),
        (2, 0.3),
        ([1, 3], [0.2, 0.4]),
        ([True, True, False, True], [0.1, 0.2, 0.4]),
    ],
    indirect=True,
)
def test_get_item(a, mask, expected):
    a = to_storch(a)
    assert (a[mask]._tensor == expected).all()


@pytest.mark.parametrize(
    "mask,value,expected",
    [
        ([False, False, True, False], [0.6], [0.1, 0.2, 0.6, 0.4]),
        ([True, True, False, True], [0.6, 0.6, 0.6], [0.6, 0.6, 0.3, 0.6]),
        (2, [0.6], [0.1, 0.2, 0.6, 0.4]),
        ([0, 1, 3], [0.6, 0.6, 0.6], [0.6, 0.6, 0.3, 0.6]),
    ],
    indirect=True,
)
def test_set_item(a, mask, value, expected):
    a = to_storch(a)
    a[mask] = value
    assert (a._tensor == expected).all()


@pytest.mark.parametrize(
    "mask_b,value,expected",
    [
        ((1, 2), 0.7, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.7]]),
        (
            torch.tensor([[True, False, True], [False, True, False]]),
            0.7,
            [[0.7, 0.2, 0.7], [0.4, 0.7, 0.6]],
        ),
    ],
    indirect=["value", "expected"],
)
def test_set_item_2(b, mask_b, value, expected):
    b = to_storch(b)
    b[mask_b] = value
    assert (b._tensor == expected).all()
