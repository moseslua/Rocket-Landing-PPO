from rocket_landing.statistics import aggregate, probability_of_improvement


def test_rliable_import_and_stratified_bootstrap_are_reproducible():
    first = aggregate(
        {"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]}, seeds=3, reps=20, random_state=5
    )
    second = aggregate(
        {"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]}, seeds=3, reps=20, random_state=5
    )
    assert first == second
    assert first["a"]["iqm"] == 2
    assert len(first["a"]["ci95"]) == 2


def test_probability_improvement_uses_paired_scores():
    result = probability_of_improvement([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], reps=20)
    assert result["probability"] == 1
