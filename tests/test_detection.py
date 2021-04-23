from fairlens.sensitive import detection as dt


def test_detect_name():
    assert dt._detect_name("Creed") == "Religion"
    assert dt._detect_name("date birth of", threshold=0.1) is None
    assert dt._detect_name("date birth of", threshold=0.5) == "Age"
    assert dt._detect_name("Sexual Preference") == "Gender"


def test_detect_names():
    cols = ["age", "gender", "legality", "risk"]
    assert dt.detect_names(cols) == ["age", "gender"]


def test_detect_names_dict():
    cols = ["age", "gender", "legality", "risk"]
    res = {"age": "Age", "gender": "Gender", "legality": None, "risk": None}
    assert dt.detect_names_dict(cols) == res
