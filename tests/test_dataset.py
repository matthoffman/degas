from .context import degas
from pathlib import Path
import os
import pandas as pd


def test_load_subset():
    path = Path(os.path.join("data", "raw", "subset.csv.gz"))
    df: pd.DataFrame = degas.dataset.load_subset(path)
    assert len(df) == 2000
    print("Columns: {}".format(df.columns))
    assert len(df.columns) == 2
    assert df.columns.contains("domain")
    assert df.columns.contains("class")
