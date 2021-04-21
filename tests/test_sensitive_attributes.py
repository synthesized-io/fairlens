from fairlens.sensitive.sensitive_attributes import detect_name, detect_names, detect_names_dict


def test_detect_name():
    assert detect_name("creed") == "Religion"
    assert detect_name("date birth of", threshold=0.1) is None
    assert detect_name("date birth of", threshold=0.5) == "Age"
    assert detect_name("sexual preference") == "Gender"


def test_detect_names():
    cols = ["age", "gender", "legality", "risk"]
    assert detect_names(cols) == ["age", "gender"]


def test_detect_names_dict():
    cols = ["age", "gender", "legality", "risk"]
    assert detect_names_dict(cols) == {"age": "Age", "gender": "Gender"}
