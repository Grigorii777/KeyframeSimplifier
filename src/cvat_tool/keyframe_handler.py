from dataclasses import dataclass
import enum
import os
from typing import Any, Optional
import json
from datetime import datetime
from pathlib import Path
import urllib3

from cvat_tool.dto import DISTANCE_FUNCTIONS, Keyframe, KeyframeField, KeyframeSimplifyingMethod, KeyframesField

from .keyframe_iou import choose_keyframes_iou

from .keyframes import choose_keyframes
import numpy as np
from cvat_sdk.core.client import Client, Config
from cvat_sdk import models
import dotenv

# Disable SSL warnings when verify_ssl=False is used
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

dotenv.load_dotenv()


class KeyframeHandler:
    def __init__(self) -> None:
        self.keyframes = []
        self.backup_dir = Path(__file__).parent / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        self.cvat_url = os.getenv("CVAT_HOST", "http://localhost:7000")
        self.username = os.getenv("CVAT_USERNAME", "admin")
        self.password = os.getenv("CVAT_PASSWORD", "password")

    def prepare_keyframes_from_shapes(self, shapes) -> list[Keyframe]:
        """Convert shapes to Keyframe objects."""
        keyframes = []
        for shape in shapes:
            kf = Keyframe(
                frame_id=shape.frame,
                position=np.array(shape.points[:3], dtype=float),
                rotation=np.array(shape.points[3:6], dtype=float),
                scale=np.array(shape.points[6:9], dtype=float),
            )
            keyframes.append(kf)
        return keyframes

    def get_simplified_frame_ids(self, shapes, fields: list[KeyframesField] = None, iou_threshold: float = 0.8, auto_percent: float = 5.0) -> set[int]:
        """Get set of frame_ids after simplification."""
        keyframes = self.prepare_keyframes_from_shapes(shapes)
        simplified = self.simplifying(keyframes=keyframes, iou_threshold=iou_threshold, fields=fields, auto_percent=auto_percent)
        return set(kf.frame_id for kf in simplified)

    def prepare_tracked_shape_requests(self, shapes, frame_ids: set[int]):
        """Convert shapes to TrackedShapeRequest for given frame_ids."""
        requests = []
        for shape in shapes:
            if shape.frame in frame_ids:
                # Convert AttributeVal to AttributeValRequest
                attributes = []
                if hasattr(shape, 'attributes') and shape.attributes:
                    for attr in shape.attributes:
                        attributes.append(
                            models.AttributeValRequest(
                                spec_id=attr.spec_id,
                                value=attr.value
                            )
                        )
                
                requests.append(
                    models.TrackedShapeRequest(
                        frame=shape.frame,
                        outside=shape.outside,
                        occluded=shape.occluded,
                        z_order=shape.z_order,
                        points=shape.points,
                        rotation=shape.rotation,
                        type=shape.type,
                        attributes=attributes
                    )
                )
        return requests

    def build_updated_track(self, track, updated_shapes):
        """Create a new LabeledTrackRequest with updated shapes."""
        # Convert AttributeVal to AttributeValRequest for track attributes
        attributes = []
        if hasattr(track, 'attributes') and track.attributes:
            for attr in track.attributes:
                attributes.append(
                    models.AttributeValRequest(
                        spec_id=attr.spec_id,
                        value=attr.value
                    )
                )
        
        return models.LabeledTrackRequest(
            frame=track.frame,
            label_id=track.label_id,
            group=track.group,
            source=track.source,
            shapes=updated_shapes,
            attributes=attributes,
            elements=track.elements if hasattr(track, 'elements') else []
        )

    def convert_shapes_to_requests(self, shapes):
        """Convert LabeledShape objects to LabeledShapeRequest objects."""
        requests = []
        for shape in shapes:
            # Convert AttributeVal to AttributeValRequest
            attributes = []
            if hasattr(shape, 'attributes') and shape.attributes:
                for attr in shape.attributes:
                    attributes.append(
                        models.AttributeValRequest(
                            spec_id=attr.spec_id,
                            value=attr.value
                        )
                    )
            
            requests.append(
                models.LabeledShapeRequest(
                    frame=shape.frame,
                    label_id=shape.label_id,
                    group=shape.group,
                    source=shape.source,
                    type=shape.type,
                    occluded=shape.occluded,
                    z_order=shape.z_order,
                    points=shape.points,
                    rotation=shape.rotation,
                    attributes=attributes,
                    elements=shape.elements if hasattr(shape, 'elements') else []
                )
            )
        return requests

    def auto_calculate_fields(self, keyframes: list[Keyframe], percent: float) -> list[KeyframesField]:
        """Auto-calculate threshold fields based on percentage of max distances."""
        if len(keyframes) < 2:
            return []
        
        fields = []
        
        # Position: percent of max distance between keyframes
        positions = np.array([kf.position for kf in keyframes])
        max_pos_dist = np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        pos_threshold = max_pos_dist * (percent / 100.0)
        fields.append(KeyframesField(
            keyframe_field=KeyframeField.POSITION,
            threshold=pos_threshold,
            method=KeyframeSimplifyingMethod.L2
        ))
        print(f"Auto-calculated position threshold: {pos_threshold:.4f} ({percent}% of max distance {max_pos_dist:.4f})")
        
        # Scale: percent of max distance between keyframes
        scales = np.array([kf.scale for kf in keyframes])
        max_scale_dist = np.max(np.linalg.norm(np.diff(scales, axis=0), axis=1))
        scale_threshold = max_scale_dist * (percent / 100.0)
        fields.append(KeyframesField(
            keyframe_field=KeyframeField.SCALE,
            threshold=scale_threshold,
            method=KeyframeSimplifyingMethod.L2
        ))
        print(f"Auto-calculated scale threshold: {scale_threshold:.4f} ({percent}% of max distance {max_scale_dist:.4f})")
        
        # Rotation: percent of range [-π, π] = 2π
        rotation_range = 2 * np.pi
        rot_threshold = rotation_range * (percent / 100.0)
        fields.append(KeyframesField(
            keyframe_field=KeyframeField.ROTATION,
            threshold=rot_threshold,
            method=KeyframeSimplifyingMethod.L_INF
        ))
        print(f"Auto-calculated rotation threshold: {rot_threshold:.4f} ({percent}% of range {rotation_range:.4f})")
        
        return fields

    def simplifying(self, keyframes: list[Keyframe], iou_threshold: float = 0.8, fields: list[KeyframesField] = None, auto_percent: float = 5.0) -> list[Keyframe]:
        # Auto-calculate fields if not provided
        if fields is None and auto_percent > 0:
            fields = self.auto_calculate_fields(keyframes, auto_percent)
        elif fields is None:
            fields = []
        
        if not fields and iou_threshold > 1.0:
            return keyframes  # No simplification needed
        
        # Start with empty set - each filter can ADD keyframes
        keyframe_indices = set()
        
        # IOU-based simplification (only if threshold <= 1.0)
        if iou_threshold <= 1.0 and iou_threshold >= 0.0:
            result = set(choose_keyframes_iou(keyframes, iou_threshold=iou_threshold))
            keyframe_indices |= result  # union
        
        # Field-based simplification
        for field in fields:
            if field.threshold > 0:
                arr = np.array([getattr(kf, field.keyframe_field.value) for kf in keyframes])
                indices = set(choose_keyframes(arr, DISTANCE_FUNCTIONS[field.method], field.threshold))
                keyframe_indices |= indices  # union
        
        return [keyframes[i] for i in sorted(keyframe_indices)]

    @staticmethod
    def _remove_ids_recursive(obj):
        """Recursively remove 'id' field from dict/list structures."""
        if isinstance(obj, dict):
            return {k: KeyframeHandler._remove_ids_recursive(v) for k, v in obj.items() if k != "id"}
        elif isinstance(obj, list):
            return [KeyframeHandler._remove_ids_recursive(item) for item in obj]
        else:
            return obj

    def _convert_annotations_to_request(self, annotations_dict: dict) -> models.LabeledDataRequest:
        """Convert annotations dict to LabeledDataRequest, removing all IDs."""
        # Remove all IDs recursively
        cleaned = self._remove_ids_recursive(annotations_dict)

        return models.LabeledDataRequest(
            version=cleaned.get("version", 0),
            tags=[models.LabeledImageRequest(**tag) for tag in cleaned.get("tags", [])],
            shapes=[models.LabeledShapeRequest(**shape) for shape in cleaned.get("shapes", [])],
            tracks=[models.LabeledTrackRequest(**track) for track in cleaned.get("tracks", [])]
        )

    def save_backup(self, annotations_data, job_id: int) -> str:
        """Save backup of annotations to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"backup_job_{job_id}_{timestamp}.json"
        backup_path = self.backup_dir / backup_filename

        # Save raw annotations dict
        backup_data = {
            "job_id": job_id,
            "timestamp": timestamp,
            "annotations": annotations_data.to_dict()
        }

        with open(backup_path, "w") as f:
            json.dump(backup_data, f, indent=2)

        print(f"✓ Backup saved: {backup_path}")
        return str(backup_path)

    def get_latest_backup(self, job_id: int) -> Optional[Path]:
        """Find the latest backup file for a given job_id."""
        backup_files = list(self.backup_dir.glob(f"backup_job_{job_id}_*.json"))
        if not backup_files:
            return None
        return max(backup_files, key=lambda p: p.stat().st_mtime)

    def restore_from_backup(self, job_id: int) -> None:
        """Restore annotations from the latest backup."""
        backup_path = self.get_latest_backup(job_id)
        if not backup_path:
            print(f"✗ No backup found for job {job_id}")
            return

        print(f"Found backup: {backup_path}")

        with open(backup_path, "r") as f:
            backup_data = json.load(f)

        print(f"Connecting to CVAT API: {self.cvat_url}")
        client = Client(url=self.cvat_url, config=Config(verify_ssl=False))
        client.login((self.username, self.password))

        try:
            print(f"Retrieving job {job_id}...")
            job = client.jobs.retrieve(job_id)

            # Reconstruct LabeledDataRequest from backup
            backup_annotations = backup_data["annotations"]
            annotations_request = self._convert_annotations_to_request(backup_annotations)

            print("Restoring annotations from backup...")
            job.set_annotations(annotations_request)
            print(f"✓ Annotations restored from backup (created at {backup_data['timestamp']})")
        finally:
            client.logout()
            print("Disconnecting from CVAT API")

    def simplifying_job(self, job_id: int, fields: list[KeyframesField] = None, iou_threshold: float = 0.8, auto_percent: float = 5.0) -> None:
        """
        Automatically downloads annotations from CVAT API, simplifies keyframes,
        and uploads updated annotations back.

        Args:
            job_id: Job ID in CVAT
            fields: List of fields with simplification settings (auto-calculated if None)
            iou_threshold: IOU threshold for simplification
            auto_percent: Percentage for auto-calculating thresholds when fields is None
        """

        print(f"Connecting to CVAT API: {self.cvat_url}")
        client = Client(url=self.cvat_url, config=Config(verify_ssl=False))
        client.login((self.username, self.password))

        len_shapes_sum = 0
        len_simplified_shapes_sum = 0

        try:
            print(f"Retrieving job {job_id}...")
            job = client.jobs.retrieve(job_id)

            print("Downloading annotations...")
            annotations_data = job.get_annotations()

            # print("Annotations downloaded:", annotations_data)

            print(f"Received tracks: {len(annotations_data.tracks)}")
            print(f"Received shapes: {len(annotations_data.shapes)}")
            print(f"Received tags: {len(annotations_data.tags)}")

            # Create backup before making changes
            self.save_backup(annotations_data, job_id)

            if not annotations_data.tracks:
                print("No tracks to process")
                return

            # Process each track
            updated_tracks = []
            for track in annotations_data.tracks:
                shapes = track.shapes
                if not shapes:
                    updated_tracks.append(track)
                    continue

                print(f"\nProcessing track {track.id} with {len(shapes)} frames")

                simplified_frame_ids = self.get_simplified_frame_ids(shapes, fields, iou_threshold, auto_percent)
                print(f"Simplified from {len(shapes)} to {len(simplified_frame_ids)} frames")
                len_shapes_sum += len(shapes)
                len_simplified_shapes_sum += len(simplified_frame_ids)
                updated_shapes = self.prepare_tracked_shape_requests(shapes, simplified_frame_ids)
                updated_track = self.build_updated_track(track, updated_shapes)
                updated_tracks.append(updated_track)

            # Convert shapes to requests
            updated_shapes = self.convert_shapes_to_requests(annotations_data.shapes)

            updated_annotations = models.LabeledDataRequest(
                version=annotations_data.version,
                tags=annotations_data.tags,
                shapes=updated_shapes,
                tracks=updated_tracks
            )

            print("\nUploading updated annotations...")
            job.set_annotations(updated_annotations)
            print("✓ Annotations successfully updated!")
            print("iou_threshold =", iou_threshold)
            print(f"Total keyframes before simplification: {len_shapes_sum}")
            print(f"Total keyframes after simplification: {len_simplified_shapes_sum}")

        finally:
            client.logout()
            print("Disconnecting from CVAT API")

    @staticmethod
    def parse_fields(fields_args):
        """
        Parse fields argument from CLI into list of KeyframesField.
        Example: --field position,5.1,l2 --field rotation,0.0,l_inf
        """
        import argparse
        fields = []
        for arg in fields_args:
            try:
                keyframe_field, threshold, method = arg.split(",")
                keyframe_field_enum = KeyframeField[keyframe_field.strip().upper()]
                # Resolve method by value (case-insensitive)
                method_value = method.strip().lower()
                try:
                    method_enum = next(m for m in KeyframeSimplifyingMethod if m.value == method_value)
                except StopIteration:
                    raise ValueError(f"Unknown method value: {method}")
                fields.append(KeyframesField(
                    keyframe_field=keyframe_field_enum,
                    threshold=float(threshold),
                    method=method_enum
                ))
            except Exception as e:
                raise argparse.ArgumentTypeError(f"Invalid --field value '{arg}': {e}")
        return fields

