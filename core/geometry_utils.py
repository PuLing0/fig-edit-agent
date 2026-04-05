"""NumPy-backed geometry and coordinate-space utilities."""

from __future__ import annotations

import numpy as np

from ..schemas.geometry import BoundingBox, CoordinateInfo, Point2D, Polygon2D


class CoordinateManager:
    """Helpers for transforming boxes and polygons between coordinate spaces."""

    @staticmethod
    def translation_to_matrix(offset_x: float, offset_y: float) -> np.ndarray:
        return np.array(
            [[1.0, 0.0, float(offset_x)], [0.0, 1.0, float(offset_y)], [0.0, 0.0, 1.0]],
            dtype=float,
        )

    @staticmethod
    def affine6_to_matrix(params: list[float]) -> np.ndarray:
        if len(params) != 6:
            raise ValueError("affine params must contain 6 values")
        a, b, c, d, e, f = params
        return np.array([[a, b, c], [d, e, f], [0.0, 0.0, 1.0]], dtype=float)

    @staticmethod
    def matrix_to_affine6(matrix: np.ndarray) -> list[float]:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError("matrix must be 3x3")
        return [
            float(matrix[0, 0]),
            float(matrix[0, 1]),
            float(matrix[0, 2]),
            float(matrix[1, 0]),
            float(matrix[1, 1]),
            float(matrix[1, 2]),
        ]

    @staticmethod
    def coordinate_info_to_matrix(coord: CoordinateInfo) -> np.ndarray:
        if coord.transform_kind == "translation_only":
            return CoordinateManager.translation_to_matrix(coord.offset_x or 0, coord.offset_y or 0)
        return CoordinateManager.affine6_to_matrix(coord.affine_matrix or [1, 0, 0, 0, 1, 0])

    @staticmethod
    def invert_matrix(matrix: np.ndarray) -> np.ndarray:
        return np.linalg.inv(np.asarray(matrix, dtype=float))

    @staticmethod
    def compose_matrices(*matrices: np.ndarray) -> np.ndarray:
        result = np.eye(3, dtype=float)
        for matrix in matrices:
            result = np.asarray(matrix, dtype=float) @ result
        return result

    @staticmethod
    def apply_matrix_to_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape [N, 2]")
        ones = np.ones((points.shape[0], 1), dtype=float)
        homogeneous = np.concatenate([points, ones], axis=1)
        transformed = (np.asarray(matrix, dtype=float) @ homogeneous.T).T
        return transformed[:, :2]

    @staticmethod
    def apply_matrix_to_point(point: Point2D, matrix: np.ndarray) -> Point2D:
        transformed = CoordinateManager.apply_matrix_to_points(
            np.array([[point.x, point.y]], dtype=float),
            matrix,
        )
        x, y = transformed[0].tolist()
        return Point2D(x=int(round(x)), y=int(round(y)))

    @staticmethod
    def bbox_to_corners(bbox: BoundingBox) -> np.ndarray:
        return np.array(
            [[bbox.left, bbox.top], [bbox.right, bbox.top], [bbox.right, bbox.bottom], [bbox.left, bbox.bottom]],
            dtype=float,
        )

    @staticmethod
    def corners_to_bbox(points: np.ndarray) -> BoundingBox:
        points = np.asarray(points, dtype=float)
        left = int(np.floor(points[:, 0].min()))
        top = int(np.floor(points[:, 1].min()))
        right = int(np.ceil(points[:, 0].max()))
        bottom = int(np.ceil(points[:, 1].max()))
        return BoundingBox(left=left, top=top, right=right, bottom=bottom)

    @staticmethod
    def apply_matrix_to_bbox(bbox: BoundingBox, matrix: np.ndarray) -> BoundingBox:
        corners = CoordinateManager.bbox_to_corners(bbox)
        transformed = CoordinateManager.apply_matrix_to_points(corners, matrix)
        return CoordinateManager.corners_to_bbox(transformed)

    @staticmethod
    def polygon_to_numpy(polygon: Polygon2D) -> np.ndarray:
        return np.array([[point.x, point.y] for point in polygon.points], dtype=float)

    @staticmethod
    def numpy_to_polygon(points: np.ndarray) -> Polygon2D:
        return Polygon2D(points=[Point2D(x=int(round(x)), y=int(round(y))) for x, y in points.tolist()])

    @staticmethod
    def apply_matrix_to_polygon(polygon: Polygon2D, matrix: np.ndarray) -> Polygon2D:
        if not polygon.points:
            return Polygon2D(points=[])
        transformed = CoordinateManager.apply_matrix_to_points(CoordinateManager.polygon_to_numpy(polygon), matrix)
        return CoordinateManager.numpy_to_polygon(transformed)

    @staticmethod
    def sync_point_between_spaces(point: Point2D, source_coord: CoordinateInfo, target_coord: CoordinateInfo) -> Point2D:
        source_to_root = CoordinateManager.coordinate_info_to_matrix(source_coord)
        target_to_root = CoordinateManager.coordinate_info_to_matrix(target_coord)
        root_to_target = CoordinateManager.invert_matrix(target_to_root)
        source_to_target = root_to_target @ source_to_root
        return CoordinateManager.apply_matrix_to_point(point, source_to_target)

    @staticmethod
    def sync_points_between_spaces(
        points: list[Point2D],
        source_coord: CoordinateInfo,
        target_coord: CoordinateInfo,
    ) -> list[Point2D]:
        if not points:
            return []
        source_to_root = CoordinateManager.coordinate_info_to_matrix(source_coord)
        target_to_root = CoordinateManager.coordinate_info_to_matrix(target_coord)
        root_to_target = CoordinateManager.invert_matrix(target_to_root)
        source_to_target = root_to_target @ source_to_root
        transformed = CoordinateManager.apply_matrix_to_points(
            np.array([[point.x, point.y] for point in points], dtype=float),
            source_to_target,
        )
        return [Point2D(x=int(round(x)), y=int(round(y))) for x, y in transformed.tolist()]

    @staticmethod
    def sync_bbox_between_spaces(bbox: BoundingBox, source_coord: CoordinateInfo, target_coord: CoordinateInfo) -> BoundingBox:
        source_to_root = CoordinateManager.coordinate_info_to_matrix(source_coord)
        target_to_root = CoordinateManager.coordinate_info_to_matrix(target_coord)
        root_to_target = CoordinateManager.invert_matrix(target_to_root)
        source_to_target = root_to_target @ source_to_root
        return CoordinateManager.apply_matrix_to_bbox(bbox, source_to_target)

    @staticmethod
    def sync_polygon_between_spaces(polygon: Polygon2D, source_coord: CoordinateInfo, target_coord: CoordinateInfo) -> Polygon2D:
        source_to_root = CoordinateManager.coordinate_info_to_matrix(source_coord)
        target_to_root = CoordinateManager.coordinate_info_to_matrix(target_coord)
        root_to_target = CoordinateManager.invert_matrix(target_to_root)
        source_to_target = root_to_target @ source_to_root
        return CoordinateManager.apply_matrix_to_polygon(polygon, source_to_target)

    @staticmethod
    def clamp_bbox(bbox: BoundingBox, width: int, height: int) -> BoundingBox:
        left = max(0, min(width - 1, bbox.left))
        top = max(0, min(height - 1, bbox.top))
        right = max(left + 1, min(width, bbox.right))
        bottom = max(top + 1, min(height, bbox.bottom))
        return BoundingBox(left=left, top=top, right=right, bottom=bottom)

    @staticmethod
    def clamp_point(point: Point2D, width: int, height: int) -> Point2D:
        return Point2D(
            x=max(0, min(width - 1, point.x)),
            y=max(0, min(height - 1, point.y)),
        )

    @staticmethod
    def clamp_polygon(polygon: Polygon2D, width: int, height: int) -> Polygon2D:
        return Polygon2D(
            points=[Point2D(x=max(0, min(width - 1, p.x)), y=max(0, min(height - 1, p.y))) for p in polygon.points]
        )

    @staticmethod
    def derive_crop_coordinate_info(
        *,
        parent_coord: CoordinateInfo,
        crop_bbox_in_parent: BoundingBox,
        parent_artifact_id: str,
    ) -> CoordinateInfo:
        new_width = crop_bbox_in_parent.right - crop_bbox_in_parent.left
        new_height = crop_bbox_in_parent.bottom - crop_bbox_in_parent.top
        if parent_coord.transform_kind == "translation_only":
            return CoordinateInfo(
                root_artifact_id=parent_coord.root_artifact_id,
                width=new_width,
                height=new_height,
                transform_kind="translation_only",
                offset_x=(parent_coord.offset_x or 0) + crop_bbox_in_parent.left,
                offset_y=(parent_coord.offset_y or 0) + crop_bbox_in_parent.top,
                parent_artifact_id=parent_artifact_id,
            )

        parent_to_root = CoordinateManager.coordinate_info_to_matrix(parent_coord)
        crop_to_parent = CoordinateManager.translation_to_matrix(crop_bbox_in_parent.left, crop_bbox_in_parent.top)
        crop_to_root = parent_to_root @ crop_to_parent
        return CoordinateInfo(
            root_artifact_id=parent_coord.root_artifact_id,
            width=new_width,
            height=new_height,
            transform_kind="affine",
            affine_matrix=CoordinateManager.matrix_to_affine6(crop_to_root),
            parent_artifact_id=parent_artifact_id,
        )


__all__ = ["CoordinateManager"]
