"""Tool-layer exports with lazy loading for heavy built-in tools."""

from __future__ import annotations

from importlib import import_module

from .base import ArtifactRegistry, BaseTool, ToolContext, ToolResult
from .registry import ToolRegistry, ensure_builtin_tools_registered, tool_registry


_LAZY_EXPORTS = {
    "CollageLayoutItem": ".image_collage",
    "CollageLayoutResult": ".image_collage",
    "CropMode": ".image_crop",
    "ImageAttribute": ".image_understand",
    "ImageCollageArgs": ".image_collage",
    "ImageCollageTool": ".image_collage",
    "ImageCropArgs": ".image_crop",
    "ImageCropTool": ".image_crop",
    "ImageEditArgs": ".image_edit",
    "ImageEditTool": ".image_edit",
    "ImageGroundingArgs": ".image_grounding",
    "ImageGroundingResult": ".image_grounding",
    "ImageGroundingTool": ".image_grounding",
    "ImageOCRArgs": ".image_ocr",
    "ImageOCRTool": ".image_ocr",
    "ImageScoreArgs": ".image_score",
    "ImageScoreResult": ".image_score",
    "ImageScoreTool": ".image_score",
    "ImageSegmentArgs": ".image_segment",
    "ImageSegmentTool": ".image_segment",
    "ImageUnderstandArgs": ".image_understand",
    "ImageUnderstandResult": ".image_understand",
    "ImageUnderstandTool": ".image_understand",
    "OCRBlock": ".image_ocr",
    "OCRResult": ".image_ocr",
    "PromptReconstructArgs": ".prompt_reconstruct",
    "PromptReconstructResult": ".prompt_reconstruct",
    "PromptReconstructTool": ".prompt_reconstruct",
    "ReferenceRewrite": ".prompt_reconstruct",
    "image_collage_tool": ".image_collage",
    "image_crop_tool": ".image_crop",
    "image_edit_tool": ".image_edit",
    "image_grounding_tool": ".image_grounding",
    "image_ocr_tool": ".image_ocr",
    "image_score_tool": ".image_score",
    "image_segment_tool": ".image_segment",
    "image_understand_tool": ".image_understand",
    "prompt_reconstruct_tool": ".prompt_reconstruct",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, package=__name__)
    return getattr(module, name)


__all__ = [
    "ArtifactRegistry",
    "BaseTool",
    "CollageLayoutItem",
    "CollageLayoutResult",
    "CropMode",
    "ImageCollageArgs",
    "ImageCollageTool",
    "ImageOCRArgs",
    "ImageOCRTool",
    "ImageScoreArgs",
    "ImageScoreResult",
    "ImageScoreTool",
    "ImageAttribute",
    "ImageCropArgs",
    "ImageCropTool",
    "ImageEditArgs",
    "ImageEditTool",
    "ImageGroundingArgs",
    "ImageGroundingResult",
    "ImageGroundingTool",
    "OCRBlock",
    "OCRResult",
    "ImageSegmentArgs",
    "ImageSegmentTool",
    "ImageUnderstandArgs",
    "ImageUnderstandResult",
    "ImageUnderstandTool",
    "PromptReconstructArgs",
    "PromptReconstructResult",
    "ReferenceRewrite",
    "PromptReconstructTool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "ensure_builtin_tools_registered",
    "image_collage_tool",
    "image_crop_tool",
    "image_edit_tool",
    "image_grounding_tool",
    "image_ocr_tool",
    "image_score_tool",
    "image_segment_tool",
    "image_understand_tool",
    "prompt_reconstruct_tool",
    "tool_registry",
]
