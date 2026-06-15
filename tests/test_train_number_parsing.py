import unittest

from backend.models.sarvam import _train_number_from_transcript


class TrainNumberParsingTests(unittest.TestCase):
    def test_hindi_script_digits(self) -> None:
        self.assertEqual(
            _train_number_from_transcript("ट्रेन एक दो एक छः चार का स्टेटस बताओ"),
            "12164",
        )

    def test_hindi_script_english_spoken_digits(self) -> None:
        self.assertEqual(
            _train_number_from_transcript("ट्रेन वन टू वन सिक्स फोर का स्टेटस बताओ"),
            "12164",
        )

    def test_romanized_hindi_compound_digits(self) -> None:
        self.assertEqual(
            _train_number_from_transcript("train baais chhe nau ek status"),
            "22691",
        )


if __name__ == "__main__":
    unittest.main()
