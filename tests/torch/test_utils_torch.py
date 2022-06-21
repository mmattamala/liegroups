import torch
from liegroups.torch import utils


def test_isclose(device):
    tol = 1e-6
    mat = torch.Tensor([0, 1, tol, 10 * tol, 0.1 * tol]).to(device)
    ans = torch.ByteTensor([1, 0, 0, 0, 1]).to(device)
    assert (utils.isclose(mat, 0., tol=tol) == ans).all()


def test_allclose(device):
    tol = 1e-6
    mat_good = torch.Tensor([0.1 * tol, 0.01 * tol, 0, 0, 0]).to(device)
    mat_bad = torch.Tensor([0, 1, tol, 10 * tol, 0.1 * tol]).to(device)
    assert utils.allclose(mat_good, 0., tol=tol)
    assert not utils.allclose(mat_bad, 0., tol=tol)


def test_outer(device):
    vec1 = torch.Tensor([1, 2, 3]).to(device)
    vec2 = torch.Tensor([0, 1, 2]).to(device)
    assert (utils.outer(vec1, vec2) == torch.mm(
        vec1.unsqueeze(dim=1), vec2.unsqueeze(dim=0))).all()

    vecs1 = torch.Tensor([[1, 2, 3], [4, 5, 6]]).to(device)
    vecs2 = torch.Tensor([[0, 1, 2], [3, 4, 5]]).to(device)
    assert (utils.outer(vecs1, vecs2) == torch.bmm(
        vecs1.unsqueeze(dim=2), vecs2.unsqueeze(dim=1))).all()


def test_trace(device):
    mat = torch.arange(1, 10).view(3, 3).to(device)
    assert utils.trace(mat)[0] == torch.trace(mat)

    mats = torch.cat([torch.arange(1, 10).view(1, 3, 3),
                      torch.arange(11, 20).view(1, 3, 3)], dim=0).to(device)
    traces = utils.trace(mats)
    assert len(traces) == 2 and \
        traces[0] == torch.trace(mats[0]) and \
        traces[1] == torch.trace(mats[1])
