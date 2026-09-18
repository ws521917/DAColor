import pytest

from dacolor.color_math import ciede2000


@pytest.mark.parametrize(
    ("lab1", "lab2", "expected"),
    [
        ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
        ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
        ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
        ((50.0, -1.3802, -84.2814), (50.0, 0.0, -82.7485), 1.0000),
        ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
    ],
)
def test_ciede2000_reference_pairs(lab1, lab2, expected):
    assert ciede2000(lab1, lab2) == pytest.approx(expected, abs=5e-4)
    assert ciede2000(lab2, lab1) == pytest.approx(expected, abs=5e-4)
