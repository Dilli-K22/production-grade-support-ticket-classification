
import json

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

import src.evaluate as evaluate


def test_evaluation_saves_metrics_and_confusion_matrix(tmp_path, monkeypatch):
    # Create a small synthetic test dataset
    test_df = pd.DataFrame(
        {
            "Document": [
                "cannot access email",
                "laptop screen broken",
                "cannot access shared drive",
                "computer screen broken",
            ],
            "Topic_group": [
                "Access",
                "Hardware",
                "Access",
                "Hardware",
            ],
        }
    )

    test_path = tmp_path / "test.csv"
    test_df.to_csv(test_path, index=False)

    # Train a lightweight deterministic model for the test
    model = Pipeline(
        [
            ("vectorizer", CountVectorizer()),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]
    )
    model.fit(
        test_df["Document"],
        test_df["Topic_group"],
    )

    model_path = tmp_path / "model.joblib"
    import joblib
    joblib.dump(model, model_path)

    reports_dir = tmp_path / "reports"

    monkeypatch.setattr(evaluate, "MODEL_PATH", model_path)
    monkeypatch.setattr(evaluate, "TEST_PATH", test_path)
    monkeypatch.setattr(evaluate, "REPORTS_DIR", reports_dir)

    # Run evaluation
    evaluate.main()

    # Verify JSON metrics
    metrics_path = reports_dir / "evaluation_metrics.json"
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["weighted_f1"] <= 1.0
    assert "classification_report" in metrics

    # Verify confusion matrix CSV
    confusion_path = reports_dir / "confusion_matrix.csv"
    assert confusion_path.exists()

    confusion_df = pd.read_csv(confusion_path, index_col="Actual")
    assert list(confusion_df.columns) == list(model.classes_)
    assert list(confusion_df.index) == list(model.classes_)
    assert int(confusion_df.to_numpy().sum()) == len(test_df)
