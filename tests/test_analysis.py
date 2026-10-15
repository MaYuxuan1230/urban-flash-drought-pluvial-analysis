"""Definition/contract tests; no observational manuscript data are used."""
from pathlib import Path
import importlib.util
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_program(name):
    spec = importlib.util.spec_from_file_location(name[:-3], ROOT / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P1 = load_program('01_event_frequency_summary.py')
P2 = load_program('02_event_response_and_metrics.py')
P3 = load_program('03_event_matching_and_effects.py')
P4 = load_program('04_gamm_response_surfaces.py')
P5 = load_program('05_city_typology.py')


def response_group(values=None):
    lags = np.arange(-4, 9)
    dates = [pd.Timestamp('2020-12-26') + pd.Timedelta(days=int(k*8)) if k <= 0
             else pd.Timestamp('2021-01-01') + pd.Timedelta(days=int((k-1)*8)) for k in lags]
    if values is None:
        values = np.r_[np.zeros(4), -1., np.repeat(-.1, 8)]
    return pd.DataFrame(dict(event_id='A', city_id='C', event_type='FD_P',
                            variable='evi_vegetation', alignment='fd', lag=lags,
                            eightday_start=dates, response_value=values,
                            response_value_physical=values))


class DefinitionTests(unittest.TestCase):
    def test_zero_event_panel_and_map_subset(self):
        events = pd.DataFrame(dict(event_id=['E'], city_id=['A'], event_type=['FD_only'], fd_onset=['2020-01-01']))
        panel = pd.DataFrame(dict(city_id=['A', 'A', 'B'], year=[2020, 2021, 2020]))
        result = P1.summarise(events, panel)
        self.assertEqual(len(result), 3)
        self.assertEqual(result.fd_count.sum(), 1)
        self.assertFalse(result.has_fd_p_in_study_period.any())
        self.assertTrue(np.isnan(result.loc[result.city_id.eq('B'), 'fd_p_fraction'].iloc[0]))
        with self.assertRaises(ValueError):
            P1.summarise(pd.concat([events, events]), panel)

    def test_short_year_end_window_uses_dates(self):
        record = P2.metric_record(response_group(), None)
        self.assertEqual(record['recovery_time_windows_after_nadir'], 1)
        self.assertEqual(record['recovery_time_days_after_nadir'], 6)
        values = np.r_[np.zeros(4), -.3, -.4, -1., np.repeat(-.1, 6)]
        record = P2.metric_record(response_group(values), None)
        self.assertEqual(record['time_to_nadir_days'], 14)

    def test_manuscript_recovery_rule_is_not_silently_changed(self):
        record = P2.metric_record(response_group(np.zeros(13)), None)
        self.assertFalse(record['nadir_below_recovery_threshold'])
        self.assertTrue(record['recovery_observed'])
        self.assertEqual(record['recovery_time_days_after_nadir'], 6)

    def test_consecutive_windows_and_censoring(self):
        lag, censored = P2.first_consecutive_lag(pd.Series([1, 3]), pd.Series([0., 0.]), 1, 'ecological')
        self.assertTrue(censored)
        self.assertTrue(np.isnan(lag))
        record = P2.metric_record(response_group(np.r_[np.zeros(4), np.repeat(-1., 9)]), None)
        self.assertTrue(record['recovery_censored'])
        self.assertTrue(np.isnan(record['recovery_time_days_after_nadir']))

    def test_trajectory_sql_and_pandas_apply_same_eligibility(self):
        a = response_group()
        b = response_group()
        b['event_id'] = 'B'
        b.loc[b.lag.eq(0), 'response_value'] = np.inf
        bad = response_group().iloc[[4]].copy()
        bad['event_id'], bad['response_value'] = 'BAD', 999.
        data = pd.concat([a, b, bad], ignore_index=True).rename(columns={
            'response_value': 'detrended_standardized_anomaly',
            'response_value_physical': 'detrended_anomaly'})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'long.parquet'
            data.to_parquet(path, index=False)
            sql = P2.lag_summary(path)
            with patch.object(P2, 'duckdb', None):
                fallback = P2.lag_summary(path)
            pd.testing.assert_frame_equal(sql.reset_index(drop=True), fallback.reset_index(drop=True), check_dtype=False)
            self.assertEqual(sql.loc[sql.lag.eq(0), 'n'].iloc[0], 1)

    def test_all_five_matching_covariates_required(self):
        base = {name: [0., 1., 2.] for name in P3.MATCH_FEATURES}
        candidates = pd.DataFrame(base)
        target = pd.Series({name: .5 for name in P3.MATCH_FEATURES})
        self.assertTrue(np.isfinite(P3.standardised_distance(target, candidates)).all())
        candidates.loc[0, P3.MATCH_FEATURES[-1]] = np.nan
        self.assertTrue(np.isinf(P3.standardised_distance(target, candidates).iloc[0]))
        self.assertEqual(len(P3.MATCH_FEATURES), 5)
        self.assertIn('post_peak_value_lag_0_4', P3.METRICS)
        self.assertNotIn('overshoot_transition_lag_1_4', P3.METRICS)

    def test_six_paired_effects_and_observed_recovery(self):
        rows, pairs = [], []
        for j in range(3):
            pairs.append(dict(pair_id=str(j), fd_p_event_id=f'T{j}', fd_only_event_id=f'C{j}'))
            for prefix in ('T', 'C'):
                for variable in ('evi_vegetation', 'gpp_primary', 'thermal_excess_day_all'):
                    row = dict(event_id=f'{prefix}{j}', variable=variable, event_response_eligible=True,
                               recovery_observed=not(prefix == 'T' and j == 2))
                    row.update({name: float(j+1) * (.2 if prefix == 'T' else .1) for name in P3.METRICS})
                    rows.append(row)
        difference = P3.matched_differences(pd.DataFrame(rows), pd.DataFrame(pairs))
        summary = P3.effect_summary(difference, 100, 12)
        self.assertEqual(len(summary), 6)
        recovery = summary.loc[summary.metric.eq('recovery_time_days_after_nadir')].iloc[0]
        self.assertEqual(recovery.n_pairs, 2)
        np.testing.assert_allclose(summary.standardized_paired_effect, summary.mean_difference / summary.sd_difference)
        np.testing.assert_allclose(summary.standardized_ci_low, summary.bootstrap_ci_low / summary.sd_difference)

    def test_gamm_excludes_invalid_events_before_counting(self):
        rows = []
        for event_type in ('FD_only', 'FD_P'):
            for j, (eligible, value) in enumerate([(True, 1), (True, 3), ('False', 99)]):
                rows.append(dict(event_id=f'{event_type}{j}', city_id=event_type, event_type=event_type,
                                 variable='gpp_primary', event_response_eligible=eligible,
                                 loss_magnitude_lag_0_4=value, longterm_aridity_index=.6,
                                 vegetation_fraction=.3, vpd_mean_drought=1., built_up_fraction=.4))
        result = P4.prepare_city_data(pd.DataFrame(rows), 'gpp_loss', 'built_up_fraction')
        self.assertTrue(result.n_events.eq(2).all())
        self.assertTrue(result.response.eq(2).all())
        self.assertEqual(P4.OUTCOMES['peak_thermal_excess']['plane'], ('vpd', 'built'))

    def test_typology_frequency_uses_full_catalogue(self):
        rows = []
        for j in range(4):
            for variable in ('evi_vegetation', 'gpp_primary', 'thermal_excess_day_all'):
                rows.append(dict(event_id=str(j), city_id='A', event_type='FD_P', variable=variable,
                                 event_response_eligible=j < 3, loss_magnitude_lag_0_4=2.*(j+1),
                                 recovery_observed=j != 1, recovery_time_days_after_nadir=[16., np.nan, 8., 99.][j],
                                 thermal_amplification_lag_0_4=float(j+1)))
        panel = pd.DataFrame(dict(city_id=['A', 'A'], year=[2020, 2021],
                                  longterm_aridity_index=.6, vegetation_fraction=.3, built_up_fraction=.4))
        catalogue = pd.DataFrame(dict(event_id=['0', '1', '2', '3'], city_id='A', event_type='FD_P', fd_onset='2020-06-01'))
        result = P5.build_city_features(pd.DataFrame(rows), panel, catalogue).iloc[0]
        self.assertEqual(result.fd_p_frequency, 2.)
        self.assertEqual(result.evi_fd_p_events, 3)
        self.assertAlmostEqual(result.evi_recovery_burden_days, (2/3)*12 + (1/3)*56)
        self.assertEqual(result.gpp_loss, 4.)

    def test_stability_uses_80_percent_without_replacement(self):
        matrix = np.arange(50*7, dtype=float).reshape(50, 7)
        seen = []
        def fake_fit(subset, clusters, *args, **kwargs):
            seen.append(subset.copy())
            return np.arange(clusters), np.arange(len(subset)) % clusters
        with patch.object(P5, 'fit_kmedoids', side_effect=fake_fit):
            P5.stability(matrix, np.arange(50) % 6, 6, 5, 1)
        self.assertTrue(all(len(x) == 40 for x in seen))
        self.assertTrue(all(len(np.unique(x[:, 0])) == 40 for x in seen))
        with patch('sys.argv', ['05_city_typology.py', '--metrics', 'm', '--city-year', 'y', '--events', 'e', '--output-dir', 'o']):
            self.assertEqual(P5.parse_args().stability_runs, 1000)

    def test_singleton_silhouette_is_zero(self):
        value = P5.silhouette_manhattan(np.array([[0.], [1.], [10.]]), np.array([0, 0, 1]))
        self.assertAlmostEqual(value, (.9 + 8/9) / 3)


if __name__ == '__main__':
    unittest.main()
