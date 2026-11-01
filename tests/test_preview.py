import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV chưa được cài")
class PreviewTests(unittest.TestCase):
    def test_preserves_ratio_with_padding(self):
        import numpy as np
        from utils.preview import fit_preview

        source = np.full((300, 400, 3), 255, dtype=np.uint8)
        for width, height, expected in (
            (1000, 600, (800, 600)),
            (400, 600, (400, 300)),
            (800, 600, (800, 600)),
            (200, 100, (133, 100)),
        ):
            with self.subTest(width=width, height=height):
                result = fit_preview(source, width, height)
                self.assertEqual(result.shape, (height, width, 3))
                ys, xs = np.where(result[:, :, 0] > 0)
                self.assertEqual((xs.max() - xs.min() + 1, ys.max() - ys.min() + 1), expected)
                self.assertLessEqual(abs((xs.min() + xs.max()) / 2 - (width - 1) / 2), 0.5)
                self.assertLessEqual(abs((ys.min() + ys.max()) / 2 - (height - 1) / 2), 0.5)
                self.assertTrue(np.all(source == 255))

    def test_rejects_empty_viewport(self):
        import numpy as np
        from utils.preview import fit_preview

        with self.assertRaises(ValueError):
            fit_preview(np.zeros((10, 10, 3), dtype=np.uint8), 0, 100)


if __name__ == "__main__":
    unittest.main()
