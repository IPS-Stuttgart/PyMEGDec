import unittest
from unittest.mock import patch

from pymegdec import stimulus_cross_subject as cross_subject
from pymegdec.stimulus_cross_subject import (
    CrossSubjectStimulusConfig,
    make_cross_subject_candidate_configs,
)


class TestCrossSubjectWindowJitter(unittest.TestCase):
    def test_candidate_configs_keep_normalized_jitter_offsets(self):
        configs = make_cross_subject_candidate_configs(
            window_centers=(0.175,),
            window_size=0.1,
            normalizations=("none",),
            window_jitter_offsets=(-0.025, 0.0, 0.025),
            components_pca_values=(float("inf"),),
        )

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].window_jitter_offsets, (0.0, -0.025, 0.025))

        jittered = cross_subject._jittered_window_configs(configs[0])
        self.assertEqual([round(config.window_center, 3) for config in jittered], [0.175, 0.15, 0.2])
        self.assertEqual({config.window_jitter_offsets for config in jittered}, {(0.0,)})

    def test_feature_cache_loads_each_jittered_window_once_per_participant(self):
        calls = []

        def fake_loader(_data_folder, participant, *, config):
            calls.append((int(participant), round(float(config.window_center), 3), config.window_jitter_offsets))
            return object()

        config = cross_subject._normalized_config(
            CrossSubjectStimulusConfig(
                window_center=0.175,
                window_size=0.1,
                normalization="none",
                window_jitter_offsets=(-0.025, 0.0, 0.025),
                components_pca=float("inf"),
            )
        )

        with patch.object(cross_subject, "load_participant_stimulus_features", side_effect=fake_loader):
            cache = cross_subject._load_feature_cache("unused", (1, 2), (config,), progress=None)

        self.assertEqual(len(cache), 3)
        self.assertEqual(
            sorted(calls),
            [
                (1, 0.15, (0.0,)),
                (1, 0.175, (0.0,)),
                (1, 0.2, (0.0,)),
                (2, 0.15, (0.0,)),
                (2, 0.175, (0.0,)),
                (2, 0.2, (0.0,)),
            ],
        )


if __name__ == "__main__":
    unittest.main()
