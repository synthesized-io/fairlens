def progress_bar_testing(i: int):
    assert isinstance(i, int)
    assert 0 <= i <= 100
    print(f"Testing Progress Bar: {i}%")
