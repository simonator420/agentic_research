import pandas as pd
import pytest

from src.data.loader import dataset_fingerprint, load_data, split_data
from src.agents.profiler import generate_profile


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "income": [30000, 50000, 70000, 90000, 110000, 130000, 150000, 170000, 190000, 210000],
        "category": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
        "target": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
    })


def test_load_data_csv(tmp_path, sample_df):
    path = tmp_path / "data.csv"
    sample_df.to_csv(path, index=False)
    loaded = load_data(str(path))
    assert loaded.shape == sample_df.shape


def test_load_data_unsupported():
    with pytest.raises(ValueError, match="Unsupported"):
        load_data("file.json")


def test_split_data(sample_df):
    X_train, X_test, y_train, y_test = split_data(sample_df, "target", test_size=0.2)
    assert len(X_train) == 8
    assert len(X_test) == 2
    assert "target" not in X_train.columns
    assert len(y_train) == len(X_train)


def test_dataset_fingerprint_shape(sample_df):
    profile = generate_profile(sample_df, "target")
    fp = dataset_fingerprint(profile)
    assert len(fp) == 8
    assert all(isinstance(v, float) for v in fp)


def test_dataset_fingerprint_values(sample_df):
    profile = generate_profile(sample_df, "target")
    fp = dataset_fingerprint(profile)
    n_rows, n_cols, n_numeric, n_categorical = fp[:4]
    assert n_rows == 10.0
    assert n_cols == 4.0
    assert n_numeric == 3.0   # age, income, target
    assert n_categorical == 1.0  # category
