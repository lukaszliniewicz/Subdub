import unittest

from subdub.cli_args import build_parser
from subdub.prompts import TRANSLATION_PROMPT_TEMPLATE


class StructureSmokeTests(unittest.TestCase):
    def test_parser_parses_core_args(self):
        parser = build_parser()
        args = parser.parse_args(["-task", "transcribe", "-i", "video.mp4", "-sl", "English"])
        self.assertEqual(args.task, "transcribe")
        self.assertEqual(args.input, "video.mp4")

    def test_parser_parses_zoom_task(self):
        parser = build_parser()
        args = parser.parse_args(["-task", "zoom-transcript", "-i", "meeting.vtt"])
        self.assertEqual(args.task, "zoom-transcript")

    def test_prompts_package_exports_translation_prompt(self):
        self.assertIn("translate machine-generated subtitles", TRANSLATION_PROMPT_TEMPLATE)


if __name__ == "__main__":
    unittest.main()
