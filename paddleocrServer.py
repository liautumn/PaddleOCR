from __future__ import annotations

import logging
import os
from typing import Any, List

from flask import Flask, jsonify, request

from paddleocr import PaddleOCR

logging.basicConfig(level=logging.INFO)

ocr = PaddleOCR(
    text_det_thresh=0.5,
    text_rec_score_thresh=0.5,
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_textline_orientation=True)

app = Flask(__name__)


def _coerce_iterable(value: Any) -> List[Any]:
    """
    Convert any iterable value (lists, tuples, numpy arrays, etc.) into
    a standard Python list while gracefully handling scalars/Nones.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return []


def _extract_list_field(prediction: object, field_name: str) -> List[Any]:
    """
    Generic extractor that searches common PaddleOCR containers for the
    provided field and returns it as a Python list.
    """
    if prediction is None:
        return []

    if isinstance(prediction, dict) and field_name in prediction:
        return _coerce_iterable(prediction.get(field_name))

    if hasattr(prediction, field_name):
        return _coerce_iterable(getattr(prediction, field_name))

    json_res = getattr(prediction, "json_res", None)
    if isinstance(json_res, dict) and field_name in json_res:
        return _coerce_iterable(json_res.get(field_name))

    payload = getattr(prediction, "__dict__", None)
    if isinstance(payload, dict) and field_name in payload:
        return _coerce_iterable(payload.get(field_name))

    return []


def _extract_rec_texts(prediction: object) -> List[str]:
    """
    Attempt to fetch the `rec_texts` list regardless of how PaddleOCR
    chooses to expose it in the current version.
    """
    rec_texts: List[str] = [
        text for text in _extract_list_field(prediction, "rec_texts") if isinstance(text, str)
    ]
    if not rec_texts:
        return []

    angles = _extract_list_field(prediction, "textline_orientation_angles")
    if not angles:
        return rec_texts

    filtered: List[str] = []
    for idx, text in enumerate(rec_texts):
        angle = angles[idx] if idx < len(angles) else None
        try:
            angle_value = float(angle) if angle is not None else None
        except (TypeError, ValueError):
            angle_value = None

        if angle_value == 1.0:
            continue

        filtered.append(text)

    return filtered


@app.post("/ocr")
def run_ocr():
    payload = request.get_json(silent=True) or {}
    urls = payload.get("urls", [])
    if not isinstance(urls, list):
        return (
            jsonify({"detail": "The 'urls' field must be a JSON array of strings."}),
            400,
        )

    urls = [str(url).strip() for url in urls if str(url).strip()]
    if not urls:
        return jsonify({"detail": "The urls list must not be empty."}), 400

    try:
        predictions = ocr.predict(input=urls)
        # for res in predictions:
        #     res.print()
        #     res.save_to_img("output")
        #     res.save_to_json("output")
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.exception("PaddleOCR prediction failed.")
        return jsonify({"detail": f"OCR failed: {exc}"}), 500

    if len(predictions) != len(urls):
        return (
            jsonify({"detail": "Mismatch between predictions and requested URLs."}),
            500,
        )

    response: List[dict] = []
    for url, prediction in zip(urls, predictions):
        rec_texts = _extract_rec_texts(prediction)
        response.append({"input_path": url, "rec_texts": rec_texts})

    return jsonify(response)


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        debug=os.getenv("FLASK_DEBUG") == "1",
    )
