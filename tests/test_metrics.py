import pandas as pd

from fairlens.bias.metrics import compute_probabilities, emd


def test_compute_probabilities():
    data = {"Name": ["Tom", "Nick", "Krish", "Tom", "John"], "Age": [20, 21, 19, 20, 18], "Score": [30, 19, 18, 13, 30]}

    df = pd.DataFrame(data)

    print(df)

    print(compute_probabilities(df, ["Name", "Age"], "Score"))
    print(emd(df, ["Name", "Age"], "Score"))
