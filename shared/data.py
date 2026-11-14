"""CITI-Senti dataset readers / task builders for citizen_feedback_split.xlsx.

Training/Validation/Testing worksheets are already finalized. This module NEVER
re-splits, reannotates, or cleans model_text. It only creates classifier inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from shared.labels import (
    ASPECT_CODES, ASPECT_NAMES, ASPECT_TO_INDEX, SENTIMENT_TO_INDEX,
    normalize_aspect, normalize_sentiment,
)

WORKBOOK_DEFAULT = Path('data/splits/citizen_feedback_split.xlsx')
TRAIN_SHEET = 'Training'
VAL_SHEET = 'Validation'
TEST_SHEET = 'Testing'

REQUIRED_COLUMNS = (
    'comment_id', 'aspect_id', 'aspect_code', 'aspect_text', 'sentiment', 'model_text'
)
OPTIONAL_COLUMNS = ('primary_language', 'languages', 'code_switched', 'text_group_id')


@dataclass(frozen=True)
class TaskData:
    texts: list[str]
    labels: np.ndarray
    row_metadata: list[dict[str, str]]


def read_annotation_sheet(workbook: str | Path, sheet: str) -> pd.DataFrame:
    """Load annotation rows from one ORIGINAL workbook worksheet.

    Does not change the workbook or modify row assignments. The text and labels
    are read from the original columns, without translation or extra cleaning.
    """
    workbook = Path(workbook)
    if not workbook.is_file():
        raise FileNotFoundError(
            f'Missing workbook: {workbook}. Put your uploaded file at '
            f'{WORKBOOK_DEFAULT} or pass --workbook with its location.'
        )
    if workbook.suffix.lower() != '.xlsx':
        raise ValueError(f'Expected an .xlsx workbook, received: {workbook.name}')
    try:
        raw = pd.read_excel(workbook, sheet_name=sheet, dtype=str, keep_default_na=False)
    except ValueError as exc:
        raise ValueError(f'Could not read worksheet {sheet!r}: {exc}') from exc

    missing = [field for field in REQUIRED_COLUMNS if field not in raw.columns]
    if missing:
        raise ValueError(
            f'{sheet}: missing {missing}. Found {list(raw.columns)}. '
            'This implementation expects the exact provided CITI-Senti split schema.'
        )
    if raw.empty:
        raise ValueError(f'{sheet}: no annotation rows found')

    frame = raw.copy()
    for field in REQUIRED_COLUMNS:
        frame[field] = frame[field].astype(str).str.strip()
        blanks = frame.index[frame[field] == ''].tolist()
        if blanks:
            examples = ', '.join(str(i + 2) for i in blanks[:5])
            raise ValueError(f'{sheet}: blank {field} on Excel row(s) {examples}')

    for field in OPTIONAL_COLUMNS:
        if field not in frame.columns:
            frame[field] = ''
        frame[field] = frame[field].astype(str).str.strip()

    # Keep the real code and verify only that it is a known taxonomy label.
    frame['aspect_code'] = frame['aspect_code'].map(normalize_aspect)
    frame['sentiment'] = frame['sentiment'].map(normalize_sentiment)
    return frame


def make_sentiment_input(model_text: str, aspect_code: str, aspect_text: str) -> str:
    """Condition on the EXACT annotated aspect target, not just its category.

    The dataset can contain different sentiments for multiple targets belonging
    to the same category in a single comment. Dropping aspect_text would create
    contradictory labels for identical inputs. At inference, a target expression
    must likewise be supplied/predicted; category detection alone is insufficient.
    """
    code = normalize_aspect(aspect_code)
    text = str(model_text).strip()
    target = str(aspect_text).strip()
    if not text or not target:
        raise ValueError('model_text and aspect_text must both be non-empty')
    return (
        f'COMMENT: {text}  ASPECT_CATEGORY: {ASPECT_NAMES[code]}'
        f'  ASPECT_TARGET: {target}'
    )


def _row_fields(row: pd.Series) -> dict[str, str]:
    return {name: str(row[name]) for name in OPTIONAL_COLUMNS}


def make_aspect_task(frame: pd.DataFrame) -> TaskData:
    """One comment -> 11 multi-hot aspect category labels."""
    texts: list[str] = []
    labels: list[np.ndarray] = []
    manifest: list[dict[str, str]] = []

    for comment_id, group in frame.groupby('comment_id', sort=False):
        versions = group['model_text'].unique()
        if len(versions) != 1:
            raise ValueError(
                f'Comment {comment_id!r} has multiple model_text values; '
                'cannot create one aspect-detection example.'
            )
        vector = np.zeros(len(ASPECT_CODES), dtype=np.int8)
        for code in group['aspect_code'].unique():
            vector[ASPECT_TO_INDEX[code]] = 1
        first = group.iloc[0]
        texts.append(str(versions[0]))
        labels.append(vector)
        manifest.append({'comment_id': str(comment_id), **_row_fields(first)})

    return TaskData(texts=texts, labels=np.stack(labels), row_metadata=manifest)


def make_sentiment_task(frame: pd.DataFrame) -> TaskData:
    """One annotated aspect row -> one target-conditioned sentiment example.

    Intentionally preserves multiple rows with the SAME comment/category but
    DIFFERENT aspect_text and sentiment (which exist in the actual dataset).
    """
    texts: list[str] = []
    labels: list[int] = []
    manifest: list[dict[str, str]] = []

    for _, row in frame.iterrows():
        code = str(row['aspect_code'])
        target = str(row['aspect_text'])
        texts.append(make_sentiment_input(str(row['model_text']), code, target))
        labels.append(SENTIMENT_TO_INDEX[str(row['sentiment'])])
        manifest.append({
            'comment_id': str(row['comment_id']),
            'aspect_id': str(row['aspect_id']),
            'aspect_code': code,
            'aspect_text': target,
            **_row_fields(row),
        })

    return TaskData(texts=texts, labels=np.asarray(labels, dtype=np.int8), row_metadata=manifest)
