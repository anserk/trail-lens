import pytest

from trail_lens.split import make_splits


class FakeDataset:
    def __len__(self) -> int:
        return 10


def test_make_splits__is_exhaustive_and_disjoint() -> None:
    splits_indexes = make_splits(
        dataset=FakeDataset(),
        val_fraction=0.1,
        test_fraction=0.1,
        seed=42,
    )

    all_indexes = splits_indexes.test + splits_indexes.train + splits_indexes.val

    assert len(splits_indexes.train) == 8
    assert len(splits_indexes.val) == 1
    assert len(splits_indexes.test) == 1
    assert len(all_indexes) == 10
    assert set(all_indexes) == set(range(10))
    assert set(splits_indexes.test).isdisjoint(splits_indexes.val)
    assert set(splits_indexes.train).isdisjoint(splits_indexes.val)
    assert set(splits_indexes.test).isdisjoint(splits_indexes.train)


def test_make_splits_allows_valid_fraction_sum() -> None:
    splits_indexes = make_splits(
        dataset=FakeDataset(),
        val_fraction=0.4,
        test_fraction=0.4,
        seed=42,
    )

    assert len(splits_indexes.train) == 2
    assert len(splits_indexes.val) == 4
    assert len(splits_indexes.test) == 4


@pytest.mark.parametrize("val_fraction", [0, 1, -0.1, 1.1])
def test_make_splits__rejects_invalid_val_fraction(val_fraction: float) -> None:
    with pytest.raises(ValueError):
        make_splits(
            dataset=FakeDataset(),
            val_fraction=val_fraction,
            test_fraction=0.2,
            seed=42,
        )


@pytest.mark.parametrize("test_fraction", [0, 1, -0.1, 1.1])
def test_make_splits__rejects_invalid_test_fraction(test_fraction: float) -> None:
    with pytest.raises(ValueError):
        make_splits(
            dataset=FakeDataset(),
            val_fraction=0.2,
            test_fraction=test_fraction,
            seed=42,
        )


@pytest.mark.parametrize(
    ("val_fraction", "test_fraction"),
    [
        (0.7, 0.4),
        (0.4, 0.7),
        (0.5, 0.5),
    ],
)
def test_make_splits__rejects_invalid_test_fraction_sum_val_fraction(
    test_fraction: float, val_fraction: float
) -> None:
    with pytest.raises(ValueError):
        make_splits(
            dataset=FakeDataset(),
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=42,
        )
