"""Run pretrained pose + image-posture models as complementary observations."""

from dataclasses import dataclass, replace
import time


@dataclass(frozen=True)
class FusionTimings:
    pose_ms: float
    posture_ms: float


def box_iou(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0., x2 - x1) * max(0., y2 - y1)
    first_area = max(0., first[2] - first[0]) * max(0., first[3] - first[1])
    second_area = max(0., second[2] - second[0]) * max(0., second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.


def fuse_people(poses, postures):
    from ai.fall_engine import PersonObservation

    candidates, overlapping_postures = [], set()
    for i, pose in enumerate(poses):
        for j, posture in enumerate(postures):
            overlap = box_iou(pose.xyxy, posture.xyxy)
            if overlap >= 0.3:
                candidates.append((overlap, i, j))
                overlapping_postures.add(j)
    candidates.sort(reverse=True)
    pose_matches, used_postures = {}, set()
    for _, i, j in candidates:
        if i not in pose_matches and j not in used_postures:
            pose_matches[i] = postures[j]
            used_postures.add(j)
    observations = []
    for i, pose in enumerate(poses):
        posture = pose_matches.get(i)
        fallen = posture.confidence if posture and posture.label.lower() == "fallen" else 0.
        observations.append(PersonObservation(pose.xyxy, pose.confidence, fallen, pose.keypoints))
    for j, posture in enumerate(postures):
        # Keep HF-only people when pose misses them. Close duplicate HF boxes
        # are not counted as additional persons, even when their label differs.
        if j in overlapping_postures:
            continue
        fallen = posture.confidence if posture.label.lower() == "fallen" else 0.
        observations.append(PersonObservation(posture.xyxy, posture.confidence, fallen, ()))
    return tuple(observations)


@dataclass
class _RecentPerson:
    box: tuple
    confident_sample: int
    attempted_sample: int = -3


def _near(first, second):
    """Conservative local association, not an identity tracker."""
    fw, fh = first[2] - first[0], first[3] - first[1]
    sw, sh = second[2] - second[0], second[3] - second[1]
    if min(fw, fh, sw, sh) <= 0 or not 0.2 <= sw * sh / (fw * fh) <= 5:
        return False
    return (box_iou(first, second) >= 0.15
            or (abs(first[0] + first[2] - second[0] - second[2]) <= fw
                and abs(first[1] + first[3] - second[1] - second[3]) <= fh))


def _matches_recent(box, person, confidence):
    return (_near(box, person.xyxy)
            and (person.confidence >= confidence or box_iou(box, person.xyxy) >= 0.3))


def _torso_quality(person):
    points = person.keypoints
    return sum(points[i][2] >= 0.5 for i in (5, 6, 11, 12)) if len(points) == 17 else 0


class FusionDetector:
    """Full-frame fusion plus optional recent-person crops using the same pose model.

    Recovery retains anchors for at most ten predict calls (~1 s at 10 Hz),
    throttles each anchor, and shares one crop-inference budget across all people.
    Call reset() between offline clips. Low pose scores remain unchanged for the
    engine to match existing tracks without lowering its identity-seeding threshold.
    """

    def __init__(self, *, confidence=0.4, image_size=640, threads=2, posture_every=1,
                 recovery=True, recovery_max_samples=10, recovery_every=3,
                 pose_backend="yolo", pose_model_dir=None,
                 posture_backend="torch", posture_model_dir=None):
        from ai.pose_detector import PoseDetector
        from ai.posture_detector import PostureDetector

        if type(posture_every) is not int or posture_every < 1:
            raise ValueError("posture_every must be a positive integer.")
        if type(recovery_max_samples) is not int or not 1 <= recovery_max_samples <= 10:
            raise ValueError("recovery_max_samples must be an integer in [1, 10].")
        if type(recovery_every) is not int or recovery_every < 1:
            raise ValueError("recovery_every must be a positive integer.")
        if pose_backend not in ("yolo", "rtmpose", "onnx", "openvino"):
            raise ValueError("Unknown pose backend.")
        if posture_backend not in ("torch", "onnx", "openvino", "ensemble"):
            raise ValueError("Unknown posture backend.")
        if pose_model_dir is not None and pose_backend not in ("onnx", "openvino"):
            raise ValueError("pose_model_dir requires an exported pose backend.")
        if posture_model_dir is not None and posture_backend not in ("onnx", "openvino", "ensemble"):
            raise ValueError("posture_model_dir requires an exported posture backend.")
        self.confidence = confidence
        self.posture_every = posture_every
        self.recovery = recovery
        self.recovery_max_samples = recovery_max_samples
        self.recovery_every = recovery_every
        pose_kwargs = dict(confidence=min(confidence, 0.2), image_size=image_size, threads=threads)
        if pose_backend == "rtmpose":
            from ai.rtmpose_detector import RTMPoseDetector
            self.pose = RTMPoseDetector(**pose_kwargs)
        elif pose_backend == "onnx":
            from ai.onnx_detector import ONNXPoseDetector
            self.pose = ONNXPoseDetector(pose_model_dir, **pose_kwargs)
        elif pose_backend == "openvino":
            from ai.openvino_detector import OpenVINOPoseDetector
            self.pose = OpenVINOPoseDetector(pose_model_dir, **pose_kwargs)
        else:
            self.pose = PoseDetector(**pose_kwargs)
        if posture_backend == "onnx":
            from ai.onnx_detector import ONNXPostureDetector
            self.posture = ONNXPostureDetector(posture_model_dir, confidence=confidence,
                                              image_size=image_size, threads=threads)
        elif posture_backend == "openvino":
            from ai.openvino_detector import OpenVINOPostureDetector
            self.posture = OpenVINOPostureDetector(posture_model_dir, confidence=confidence,
                                                  image_size=image_size, threads=threads)
        elif posture_backend == "ensemble":
            from ai.posture_ensemble import EnsemblePostureDetector
            self.posture = EnsemblePostureDetector(confidence=confidence,
                                                   image_size=image_size, threads=threads,
                                                   backend="openvino" if posture_model_dir else "torch",
                                                   model_dir=posture_model_dir)
        else:
            self.posture = PostureDetector(confidence=confidence, image_size=image_size, threads=threads)
        self.reset()

    def reset(self):
        """Clear crop anchors and cadence before a new clip; retain loaded models."""
        self._sample = 0
        self._recent = []
        self.timings = FusionTimings(0., 0.)

    def _recover(self, image, poses):
        # Anchors are actual confident pose observations, never padded crop bounds.
        # Weak observations neither move anchors nor extend their confidence lifetime.
        self._recent = [r for r in self._recent
                        if self._sample - r.confident_sample <= self.recovery_max_samples]
        poses = list(poses)
        candidates, used = [], set()
        for recent in self._recent:
            matches = [i for i, p in enumerate(poses)
                       if i not in used and _matches_recent(recent.box, p, self.confidence)]
            index = max(matches, key=lambda i: box_iou(recent.box, poses[i].xyxy)) if matches else None
            if index is not None:
                used.add(index)
            if ((index is None or poses[index].confidence < self.confidence
                 or _torso_quality(poses[index]) < 4)
                    and self._sample - recent.attempted_sample >= self.recovery_every):
                candidates.append((recent, index))
        # Oldest attempt first prevents a missing person monopolizing the one-crop budget.
        candidates.sort(key=lambda item: item[0].attempted_sample)
        height, width = image.shape[:2]
        for recent, index in candidates:
            recent.attempted_sample = self._sample
            x1, y1, x2, y2 = recent.box
            half_w, half_h = max(32., (x2 - x1) * 0.9), max(32., (y2 - y1) * 0.9)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bounds = (max(0., cx - half_w), max(0., cy - half_h),
                      min(width, cx + half_w), min(height, cy + half_h))
            # A local zoom only: do not run a second near-whole-frame inference.
            if (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) > width * height * 0.6:
                continue
            recovered = [p for p in self.pose.predict_crop(image, bounds)
                         if _matches_recent(recent.box, p, self.confidence)]
            recovered.sort(key=lambda p: (_torso_quality(p), p.confidence), reverse=True)
            for person in recovered:
                duplicates = [i for i, p in enumerate(poses) if _near(person.xyxy, p.xyxy)]
                if index is None:
                    if not duplicates:
                        poses.append(person)
                        break
                elif duplicates == [index]:
                    # Replace one weak observation, never append a second box for it.
                    current = poses[index]
                    if ((_torso_quality(person), person.confidence)
                            > (_torso_quality(current), current.confidence)):
                        poses[index] = person
                        break
            break  # At most one additional inference, even when the crop finds nothing.
        used = set()
        for person in poses:
            matches = [i for i, r in enumerate(self._recent)
                       if i not in used and _matches_recent(r.box, person, self.confidence)]
            index = max(matches, key=lambda i: box_iou(self._recent[i].box, person.xyxy)) if matches else None
            if index is not None:
                recent = self._recent[index]
                if person.confidence >= self.confidence:
                    recent.box = person.xyxy
                    recent.confident_sample = self._sample
                used.add(index)
            elif person.confidence >= self.confidence and len(self._recent) < 20:
                self._recent.append(_RecentPerson(person.xyxy, self._sample, self._sample - self.recovery_every))
                used.add(len(self._recent) - 1)
        return tuple(poses)

    def predict(self, image):
        started = time.perf_counter()
        poses = self.pose.predict(image)
        # __new__-constructed legacy callers may not have enabled recovery.
        if getattr(self, "recovery", False):
            poses = self._recover(image, poses)
        pose_done = time.perf_counter()
        run_posture = not poses or self._sample % self.posture_every == 0
        self._sample += 1
        postures = self.posture.predict(image) if run_posture else ()
        completed = time.perf_counter()
        self.timings = FusionTimings((pose_done - started) * 1000,
                                    (completed - pose_done) * 1000 if run_posture else 0.)
        observations = fuse_people(poses, postures)
        return observations if run_posture else tuple(replace(p, posture_observed=False) for p in observations)
