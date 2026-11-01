import importlib.util
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV chưa được cài")
class FallPreviewTests(unittest.TestCase):
    def test_risk_box_is_red_and_scaled_to_preview(self):
        import numpy as np
        from utils.fall_preview import draw_tracks

        image = np.zeros((200, 300, 3), dtype=np.uint8)
        track = SimpleNamespace(xyxy=(40, 60, 200, 240), state="DA NGA",
                                event_id="fall-1", track_id=1)
        draw_tracks(image, (track,), (400, 600, 3))
        self.assertEqual(tuple(image[50, 20]), (0, 0, 255))

    def test_hidden_confirmed_event_stays_red_but_suspicion_does_not(self):
        import numpy as np
        from utils.fall_preview import draw_tracks

        for confirmed in (False, True):
            image = np.zeros((200, 300, 3), dtype=np.uint8)
            track = SimpleNamespace(xyxy=(40, 60, 200, 240), state="MAT DAU",
                                    event_id="fall-1", track_id=1, visible=False,
                                    fall_detected=confirmed)
            draw_tracks(image, (track,), (400, 600, 3))
            red = np.any((image[:, :, 2] > 0) & (image[:, :, 0] == 0) & (image[:, :, 1] == 0))
            self.assertEqual(bool(red), confirmed)

    def test_partial_visible_person_keeps_neutral_box(self):
        import numpy as np
        from utils.fall_preview import draw_tracks

        image = np.zeros((200, 300, 3), dtype=np.uint8)
        track = SimpleNamespace(xyxy=(40, 60, 200, 240), state="MAT DAU",
                                event_id=None, track_id=1, visible=True)
        draw_tracks(image, (track,), (400, 600, 3))
        self.assertEqual(tuple(image[50, 20]), (240, 240, 240))

    def test_limited_or_missing_observation_is_not_labelled_no_fall(self):
        from unittest.mock import patch
        import numpy as np
        from utils.fall_preview import draw_tracks

        for visible in (False, True):
            track = SimpleNamespace(xyxy=(40, 60, 200, 240), state="MAT DAU",
                                    track_id=1, visible=visible, observation_quality=False,
                                    fall_detected=False)
            with patch("utils.fall_preview.cv2.putText") as text:
                draw_tracks(np.zeros((200, 300, 3), dtype=np.uint8), (track,), (400, 600, 3))
            label = text.call_args.args[1]
            self.assertIn("QUAN SAT KHONG RO", label)
            self.assertNotIn("KHONG PHAT HIEN NGA", label)

    def test_no_red_box_without_fall_risk(self):
        import numpy as np
        from utils.fall_preview import draw_tracks

        for state in ("BINH THUONG", "MAT DAU", "NGHI NGA"):
            with self.subTest(state=state):
                image = np.zeros((200, 300, 3), dtype=np.uint8)
                track = SimpleNamespace(xyxy=(40, 60, 200, 240), state=state,
                                        event_id=None, track_id=1)
                draw_tracks(image, (track,), (400, 600, 3))
                self.assertFalse(np.any((image[:, :, 2] > 0) & (image[:, :, 0] == 0) & (image[:, :, 1] == 0)))


if __name__ == "__main__":
    unittest.main()
