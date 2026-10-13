import numpy as np
import pytest

robjects = pytest.importorskip('rpy2.robjects')
from loveslide import Knockoffs, OptimizeSLIDE, init_data


@pytest.mark.parametrize('method', [
    Knockoffs.filter_knockoffs_iterative,
    Knockoffs.select_short_freq,
    OptimizeSLIDE.__new__(OptimizeSLIDE).run_pipeline,
])
def test_worker_argument_is_rejected(method):
    with pytest.raises(TypeError, match='n_workers'):
        method(n_workers=2)


def test_worker_configuration_is_rejected():
    with pytest.raises(ValueError, match='knockoffs run serially'):
        init_data({'n_workers': 2})


def test_serial_selection_matches_reference_fixture_and_repeats():
    if not robjects.r('isTRUE(requireNamespace("knockoff", quietly=TRUE))')[0]:
        pytest.skip('R knockoff package is required')
    rng = np.random.default_rng(104)
    z = rng.normal(size=(180, 12))
    y = rng.normal(size=180) * .5
    y += 2 * z[:, 0] + z[:, 1]
    for _ in range(2):
        robjects.r('set.seed(1004)')
        selected = Knockoffs.select_short_freq(
            z, y, niter=30, spec=.2, fdr=.1, f_size=12)
        # Original R implementation's selection for this saved fixture.
        np.testing.assert_array_equal(selected, [0, 1])
