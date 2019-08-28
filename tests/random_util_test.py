from adam.random_utils import RotatingIndexChooser


def test_rotating_index_chooser():
    items = ["a", "b", "c"]
    rotating_chooser = RotatingIndexChooser()
    assert ("a", "b", "c", "a") == tuple(rotating_chooser.choice(items) for _ in range(4))
