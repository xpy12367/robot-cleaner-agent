import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_FILE = PROJECT_ROOT / "app.py"


class StreamlitSmokeTest(unittest.TestCase):
    def run_app(self) -> AppTest:
        app = AppTest.from_file(str(APP_FILE))
        app.run(timeout=20)
        self.assertFalse(app.exception, app.exception)
        return app

    def test_app_renders_connection_fields(self):
        app = self.run_app()

        labels = [widget.label for widget in app.text_input]
        self.assertEqual(labels[:2], ["API Key", "Base URL"])
        self.assertTrue(app.chat_input)

    def test_invalid_base_url_is_rejected(self):
        app = self.run_app()

        app.text_input[1].set_value("not-a-valid-url")
        app.button[0].click()
        app.run(timeout=20)

        self.assertFalse(app.exception, app.exception)
        self.assertTrue(
            any("Base URL" in error.value for error in app.error),
            [error.value for error in app.error],
        )


if __name__ == "__main__":
    unittest.main()
