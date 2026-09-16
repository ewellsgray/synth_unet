import pytest

from data.real_dataset import RealTissuePatchDataset


def test_instantiation_raises_not_implemented(tmp_path):
    # __init__ eagerly calls _load_index(), which is an intentional stub —
    # this test documents that contract so it's noticed if/when the stub
    # is ever implemented for real.
    with pytest.raises(NotImplementedError):
        RealTissuePatchDataset(index_path=str(tmp_path / "index.csv"), patch_size=64)
