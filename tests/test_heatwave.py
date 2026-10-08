import unittest
import tempfile
import numpy as np
from pathlib import Path
from heatwave import synthetic, validate, NeuroFuzzy, train, predict_file, FEATURES
class Tests(unittest.TestCase):
    def test_stable_firing(self):
        model = NeuroFuzzy(np.zeros((2, 5)), np.ones((2, 5)))
        weights = model.firing(np.full((3, 5), 10000.))
        self.assertTrue(np.isfinite(weights).all())
        np.testing.assert_allclose(weights.sum(axis=1), 1)
    def test_reject_invalid_input(self):
        data = synthetic(100); data.loc[0, 'relative_humidity_pct'] = 101
        with self.assertRaises(ValueError): validate(data)
    def test_round_trip(self):
        data = synthetic(240)
        with tempfile.TemporaryDirectory() as tmp:
            reports = train(data, tmp, rules=3, population=4, generations=2)
            result = predict_file(Path(tmp) / 'model.pkl', data[FEATURES].iloc[:5])
            self.assertEqual(len(result), 5)
            np.testing.assert_allclose(result.filter(like='probability_').sum(axis=1), 1)
            self.assertEqual(len(reports), 4)
            self.assertTrue((Path(tmp) / 'evaluation.png').exists())
if __name__ == '__main__': unittest.main()
