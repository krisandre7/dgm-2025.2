from src.samplers.finite import FiniteSampler


def test_finite_sampler_returns_requested_count_above_dataset_size():
    dataset = list(range(95))
    sampler = FiniteSampler(dataset, 500, allow_replacement=True)

    indices = list(iter(sampler))

    assert len(sampler) == 500
    assert len(indices) == 500
    assert all(0 <= index < len(dataset) for index in indices)
    assert set(indices[: len(dataset)]) == set(dataset)


def test_finite_sampler_samples_without_replacement_within_dataset_size():
    dataset = list(range(95))
    sampler = FiniteSampler(dataset, 50)

    indices = list(iter(sampler))

    assert len(indices) == 50
    assert len(set(indices)) == 50


def test_finite_sampler_caps_to_dataset_size_by_default():
    dataset = list(range(95))
    sampler = FiniteSampler(dataset, 500)

    indices = list(iter(sampler))

    assert len(sampler) == 95
    assert len(indices) == 95
    assert set(indices) == set(dataset)
