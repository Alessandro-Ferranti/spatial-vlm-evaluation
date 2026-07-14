import unittest

from spatial_vlm_eval.metrics import bbox_iou, normalize_relation_answer, parse_bbox_from_text


class MetricsTest(unittest.TestCase):
    def test_normalize_relation_answer(self):
        self.assertEqual(normalize_relation_answer("It is to the left of the square."), "left")
        self.assertEqual(normalize_relation_answer("BELOW"), "below")

    def test_parse_bbox_from_text(self):
        self.assertEqual(parse_bbox_from_text("[0.1, 0.2, 0.3, 0.4]"), [0.1, 0.2, 0.3, 0.4])
        self.assertIsNone(parse_bbox_from_text("no box"))

    def test_bbox_iou(self):
        self.assertEqual(bbox_iou([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)
        self.assertEqual(bbox_iou([0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
