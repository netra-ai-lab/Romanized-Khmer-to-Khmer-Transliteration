import unittest

from khmer_transliterator import transliterate, transliterate_top_n, transliterate_with_dict


class TestTransliterate(unittest.TestCase):

    def test_returns_string(self):
        result = transliterate("sokha")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_empty_input_returns_empty_string(self):
        self.assertEqual(transliterate(""), "")
        self.assertEqual(transliterate("   "), "")


class TestTransliterateTopN(unittest.TestCase):

    def test_returns_list_of_n_strings(self):
        result = transliterate_top_n("sokha", n=3)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        for item in result:
            self.assertIsInstance(item, str)

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(transliterate_top_n(""), [])


class TestTransliterateWithDict(unittest.TestCase):

    def test_returns_list(self):
        result = transliterate_with_dict("sokha", n=5)
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), 5)
        for word in result:
            self.assertIsInstance(word, str)
            self.assertGreater(len(word), 0)

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(transliterate_with_dict(""), [])

    def test_returns_fewer_than_n_when_no_matches(self):
        # Nonsense input likely has few or no valid dictionary matches
        result = transliterate_with_dict("zzzzzzz", n=5)
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), 5)


if __name__ == "__main__":
    unittest.main()
