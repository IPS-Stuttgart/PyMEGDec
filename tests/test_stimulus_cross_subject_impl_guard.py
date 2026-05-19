import os
import subprocess
import sys
import textwrap
import unittest


class TestStimulusCrossSubjectImplGuard(unittest.TestCase):
    def test_public_facade_still_imports(self):
        code = """
        from pymegdec.stimulus_cross_subject import CrossSubjectStimulusConfig

        config = CrossSubjectStimulusConfig()
        assert config.trial_selection == "random"
        """
        self._run_python(code)

    def test_private_impl_reload_is_blocked(self):
        code = """
        import importlib

        import pymegdec.stimulus_cross_subject
        import pymegdec._stimulus_cross_subject_impl as impl

        try:
            importlib.reload(impl)
        except ImportError as exc:
            message = str(exc)
            assert "internal unpatched implementation" in message, message
            assert "stimulus_cross_subject" in message, message
        else:
            raise AssertionError("implementation reload unexpectedly succeeded")
        """
        self._run_python(code)

    def test_private_core_import_exposes_patched_trial_selection(self):
        code = """
        from pymegdec import _stimulus_cross_subject_core as core

        config = core.CrossSubjectStimulusConfig()
        assert config.trial_selection == "random"
        assert hasattr(core.ParticipantFeatureSet, "trial_indices")
        """
        self._run_python(code)

    def _run_python(self, code):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(sys.path) + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
