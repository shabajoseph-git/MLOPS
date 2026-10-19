import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.train import make_pipeline


def test_pipeline_handles_missing_values_and_unknown_categories():
    X = pd.DataFrame({
        "age": [40.0, None, 60.0, 50.0],
        "sex": [0, 1, 1, 0],
        "cp": [1, 2, 3, 1],
    })
    y = [0, 1, 1, 0]
    model = make_pipeline(X, LogisticRegression(max_iter=1000))
    model.fit(X, y)
    pred = model.predict(pd.DataFrame({"age": [45.0], "sex": [1], "cp": [99]}))
    assert len(pred) == 1
    assert int(pred[0]) in (0, 1)
