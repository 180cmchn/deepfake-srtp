import os
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime
from unittest.mock import AsyncMock, patch

from app.schemas.detection import DetectionRequest, VideoDetectionRequest
from app.services.detection_service import DetectionService


class DummyBackgroundTasks:
    def __init__(self):
        self.tasks = []

    def add_task(self, func, *args, **kwargs):
        self.tasks.append((func, args, kwargs))


@dataclass
class SavedDetectionRecord:
    id: int
    created_at: datetime


class DetectionServiceDecisionSemanticsTests(unittest.IsolatedAsyncioTestCase):
    def assert_threshold_semantics(
        self,
        *,
        prediction,
        confidence,
        decision_metrics,
        probabilities,
        threshold,
    ):
        prediction_value = (
            prediction.value if hasattr(prediction, "value") else prediction
        )
        expected_prediction = "fake" if probabilities["fake"] >= threshold else "real"
        expected_confidence = probabilities[expected_prediction]
        opposite_class = "real" if expected_prediction == "fake" else "fake"

        self.assertEqual(prediction_value, expected_prediction)
        self.assertAlmostEqual(confidence, expected_confidence)
        self.assertAlmostEqual(
            decision_metrics.confidence_threshold,
            threshold,
        )
        self.assertAlmostEqual(
            decision_metrics.fake_probability,
            probabilities["fake"],
        )
        self.assertAlmostEqual(
            decision_metrics.real_probability,
            probabilities["real"],
        )
        self.assertAlmostEqual(
            decision_metrics.predicted_probability,
            expected_confidence,
        )
        self.assertAlmostEqual(
            decision_metrics.decision_margin,
            expected_confidence - probabilities[opposite_class],
        )
        self.assertAlmostEqual(
            decision_metrics.threshold_gap,
            probabilities["fake"] - threshold,
        )
        self.assertTrue(decision_metrics.threshold_applied_to_fake)
        if probabilities[expected_prediction] != probabilities[opposite_class]:
            self.assertNotAlmostEqual(
                confidence,
                probabilities[opposite_class],
            )
            self.assertNotAlmostEqual(
                decision_metrics.predicted_probability,
                probabilities[opposite_class],
            )

    async def run_detect_file_case(self, probabilities, threshold):
        service = DetectionService(db=None)
        loaded_model = {
            "model": object(),
            "model_type": "vit",
            "input_size": 224,
            "source": "registry",
        }
        saved_record = SavedDetectionRecord(id=99, created_at=datetime.now())

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(b"image-bytes")
            file_path = temp_file.name

        try:
            with (
                patch.object(
                    service,
                    "_load_model",
                    new=AsyncMock(return_value=loaded_model),
                ),
                patch.object(
                    service,
                    "_preprocess_image",
                    return_value=object(),
                ),
                patch.object(
                    service,
                    "_predict_tensor",
                    new=AsyncMock(return_value={"probabilities": probabilities}),
                ),
                patch.object(
                    service,
                    "_save_detection_result",
                    new=AsyncMock(return_value=saved_record),
                ),
            ):
                return await service.detect_file(
                    file_path=file_path,
                    request=DetectionRequest(
                        model_type="vit",
                        confidence_threshold=threshold,
                        return_probabilities=True,
                    ),
                    background_tasks=DummyBackgroundTasks(),
                )
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_probability_helpers_keep_threshold_and_class_confidence_aligned(self):
        service = DetectionService(db=None)
        cases = (
            {
                "name": "threshold_zero_equal_boundary",
                "probabilities": {"fake": 0.0, "real": 1.0},
                "threshold": 0.0,
            },
            {
                "name": "threshold_one_equal_boundary",
                "probabilities": {"fake": 1.0, "real": 0.0},
                "threshold": 1.0,
            },
            {
                "name": "just_below_threshold",
                "probabilities": {"fake": 0.7999, "real": 0.2001},
                "threshold": 0.8,
            },
            {
                "name": "equal_threshold",
                "probabilities": {"fake": 0.8, "real": 0.2},
                "threshold": 0.8,
            },
            {
                "name": "just_above_threshold",
                "probabilities": {"fake": 0.8001, "real": 0.1999},
                "threshold": 0.8,
            },
        )

        for case in cases:
            with self.subTest(case=case["name"]):
                prediction, confidence = service._probabilities_to_prediction(
                    case["probabilities"],
                    confidence_threshold=case["threshold"],
                )
                decision_metrics = service._build_decision_metrics(
                    case["probabilities"],
                    case["threshold"],
                    prediction,
                    confidence,
                )

                self.assert_threshold_semantics(
                    prediction=prediction,
                    confidence=confidence,
                    decision_metrics=decision_metrics,
                    probabilities=case["probabilities"],
                    threshold=case["threshold"],
                )

    async def test_detect_file_threshold_edges_keep_decision_metrics_non_contradictory(
        self,
    ):
        cases = (
            {
                "name": "threshold_zero_equal_boundary",
                "probabilities": {"fake": 0.0, "real": 1.0},
                "threshold": 0.0,
            },
            {
                "name": "threshold_one_equal_boundary",
                "probabilities": {"fake": 1.0, "real": 0.0},
                "threshold": 1.0,
            },
            {
                "name": "just_below_threshold",
                "probabilities": {"fake": 0.7999, "real": 0.2001},
                "threshold": 0.8,
            },
            {
                "name": "equal_threshold",
                "probabilities": {"fake": 0.8, "real": 0.2},
                "threshold": 0.8,
            },
            {
                "name": "just_above_threshold",
                "probabilities": {"fake": 0.8001, "real": 0.1999},
                "threshold": 0.8,
            },
        )

        for case in cases:
            with self.subTest(case=case["name"]):
                response = await self.run_detect_file_case(
                    case["probabilities"],
                    case["threshold"],
                )

                self.assertTrue(response.success)
                result = response.result
                self.assertIsNotNone(result)
                if result is None:
                    self.fail(
                        "detect_file should return a result for a successful boundary case"
                    )

                self.assertAlmostEqual(
                    result.probabilities["fake"],
                    case["probabilities"]["fake"],
                )
                self.assertAlmostEqual(
                    result.probabilities["real"],
                    case["probabilities"]["real"],
                )
                self.assert_threshold_semantics(
                    prediction=result.prediction,
                    confidence=result.confidence,
                    decision_metrics=result.decision_metrics,
                    probabilities=case["probabilities"],
                    threshold=case["threshold"],
                )

    async def test_detect_file_reports_confidence_for_returned_class(self):
        service = DetectionService(db=None)
        loaded_model = {
            "model": object(),
            "model_type": "vit",
            "input_size": 224,
            "source": "registry",
        }
        saved_record = SavedDetectionRecord(id=1, created_at=datetime.now())

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(b"image-bytes")
            file_path = temp_file.name

        try:
            with (
                patch.object(
                    service,
                    "_load_model",
                    new=AsyncMock(return_value=loaded_model),
                ),
                patch.object(
                    service,
                    "_preprocess_image",
                    return_value=object(),
                ),
                patch.object(
                    service,
                    "_predict_tensor",
                    new=AsyncMock(
                        return_value={"probabilities": {"fake": 0.8, "real": 0.2}}
                    ),
                ),
                patch.object(
                    service,
                    "_save_detection_result",
                    new=AsyncMock(return_value=saved_record),
                ),
            ):
                response = await service.detect_file(
                    file_path=file_path,
                    request=DetectionRequest(
                        model_type="vit",
                        confidence_threshold=0.9,
                        return_probabilities=True,
                    ),
                    background_tasks=DummyBackgroundTasks(),
                )
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

        self.assertTrue(response.success)
        result = response.result
        self.assertIsNotNone(result)
        if result is None:
            self.fail("detect_file should return a result for a successful response")

        self.assertEqual(result.prediction, "real")
        self.assertAlmostEqual(result.confidence, 0.2)
        self.assertAlmostEqual(result.probabilities["fake"], 0.8)
        self.assertAlmostEqual(result.probabilities["real"], 0.2)
        self.assertAlmostEqual(
            result.decision_metrics.predicted_probability,
            result.confidence,
        )
        self.assertAlmostEqual(result.decision_metrics.threshold_gap, -0.1)
        self.assertTrue(result.decision_metrics.threshold_applied_to_fake)

    async def test_detect_video_uses_same_thresholded_confidence_semantics(self):
        service = DetectionService(db=None)
        loaded_model = {
            "model": object(),
            "model_type": "vit",
            "input_size": 224,
            "source": "registry",
        }
        saved_record = SavedDetectionRecord(id=2, created_at=datetime.now())

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(b"video-bytes")
            file_path = temp_file.name

        try:
            with (
                patch.object(
                    service,
                    "_extract_frames",
                    new=AsyncMock(return_value=[(object(), 0.0), (object(), 1.0)]),
                ),
                patch.object(
                    service,
                    "_load_model",
                    new=AsyncMock(return_value=loaded_model),
                ),
                patch.object(
                    service,
                    "_preprocess_frame",
                    return_value=object(),
                ),
                patch.object(
                    service,
                    "_predict_tensor",
                    new=AsyncMock(
                        side_effect=[
                            {"probabilities": {"fake": 0.8, "real": 0.2}},
                            {"probabilities": {"fake": 0.8, "real": 0.2}},
                        ]
                    ),
                ),
                patch.object(
                    service,
                    "_save_detection_result",
                    new=AsyncMock(return_value=saved_record),
                ),
            ):
                response = await service.detect_video(
                    video_path=file_path,
                    request=VideoDetectionRequest(
                        model_type="vit",
                        confidence_threshold=0.9,
                        return_frame_results=True,
                    ),
                    background_tasks=DummyBackgroundTasks(),
                )
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

        self.assertTrue(response.success)
        aggregated_result = response.aggregated_result
        self.assertIsNotNone(aggregated_result)
        if aggregated_result is None:
            self.fail("detect_video should return an aggregated result by default")

        self.assertEqual(aggregated_result.prediction, "real")
        self.assertAlmostEqual(aggregated_result.confidence, 0.2)
        self.assertAlmostEqual(aggregated_result.probabilities["fake"], 0.8)
        self.assertAlmostEqual(aggregated_result.probabilities["real"], 0.2)
        self.assertAlmostEqual(
            aggregated_result.decision_metrics.predicted_probability,
            aggregated_result.confidence,
        )

        frame_results = response.frame_results
        self.assertIsNotNone(frame_results)
        if frame_results is None:
            self.fail("detect_video should return frame results when requested")

        self.assertEqual(len(frame_results), 2)
        for frame_result in frame_results:
            self.assertEqual(frame_result.result.prediction, "real")
            self.assertAlmostEqual(frame_result.result.confidence, 0.2)
            self.assertAlmostEqual(
                frame_result.result.decision_metrics.predicted_probability,
                frame_result.result.confidence,
            )


if __name__ == "__main__":
    unittest.main()
