import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from goldminer.env import load_dotenv
from goldminer.errors import GoldMinerError


class DotenvTests(unittest.TestCase):
    def test_loads_values_comments_and_quotes(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {}, clear=True):
            path = Path(td) / ".env"
            path.write_text(
                '# comment\nOPENAI_API_KEY="secret-value"\nOPENAI_TRANSCRIPTION_MODEL=model # note\n'
            )
            self.assertTrue(load_dotenv(path))
            self.assertEqual(os.environ["OPENAI_API_KEY"], "secret-value")
            self.assertEqual(os.environ["OPENAI_TRANSCRIPTION_MODEL"], "model")

    def test_shell_value_takes_precedence(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(
            os.environ, {"OPENAI_API_KEY": "from-shell"}, clear=True
        ):
            path = Path(td) / ".env"
            path.write_text("OPENAI_API_KEY=from-file\n")
            load_dotenv(path)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "from-shell")

    def test_missing_file_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(load_dotenv(Path(td) / ".env"))

    def test_malformed_line_is_actionable(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / ".env"
            path.write_text("NOT_AN_ASSIGNMENT\n")
            with self.assertRaisesRegex(GoldMinerError, "line 1"):
                load_dotenv(path)


if __name__ == "__main__":
    unittest.main()
