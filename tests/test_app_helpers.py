import os
import tempfile
import unittest

import litellm

from subdub.ai.client import configure_litellm_callbacks, litellm_success_callback
from subdub.app_helpers import apply_default_align_model, build_provider_params, prepare_context_from_input
from subdub.cli_args import build_parser


class AppHelpersTests(unittest.TestCase):
    def test_apply_default_align_model_sets_expected_model(self):
        parser = build_parser()
        args = parser.parse_args(["-i", "input.wav", "-sl", "English"])
        args.align_model = None

        apply_default_align_model(args)

        self.assertEqual(args.align_model, "jonatasgrosman/wav2vec2-xls-r-1b-english")

    def test_build_provider_params_sets_local_api_key_fallback(self):
        parser = build_parser()
        args = parser.parse_args([
            "-i",
            "input.wav",
            "-api_base",
            "http://127.0.0.1:1234/v1",
            "-model",
            "openai/gpt-4o-mini",
        ])

        original_openai_key = os.environ.get("OPENAI_API_KEY")
        try:
            os.environ.pop("OPENAI_API_KEY", None)
            provider_params = build_provider_params(args)
            self.assertEqual(provider_params.get("api_key"), "lm-studio")
            self.assertEqual(os.environ.get("OPENAI_API_KEY"), "lm-studio")
        finally:
            if original_openai_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = original_openai_key

    def test_prepare_context_from_input_local_file(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "demo_input.wav")
            with open(input_path, "wb") as fh:
                fh.write(b"stub")

            custom_session = os.path.join(tmpdir, "session_a")
            args = parser.parse_args(["-i", input_path, "-session", custom_session])

            context, session_folder, video_path, video_name = prepare_context_from_input(args)

            self.assertEqual(video_path, os.path.abspath(input_path))
            self.assertEqual(video_name, "demo_input")
            self.assertTrue(os.path.isdir(session_folder))
            self.assertEqual(session_folder, os.path.abspath(custom_session))
            self.assertEqual(context.video_path, os.path.abspath(input_path))

    def test_configure_litellm_callbacks_is_idempotent(self):
        configure_litellm_callbacks()
        configure_litellm_callbacks()
        self.assertIsInstance(litellm.success_callback, list)
        self.assertIn(litellm_success_callback, litellm.success_callback)


if __name__ == "__main__":
    unittest.main()
