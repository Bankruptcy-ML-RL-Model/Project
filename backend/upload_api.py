"""
Upload API — Accepts 6 PDF financial statements and returns the 48-ratio feature vector.
"""

import os
import tempfile
import shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from backend.financial_extractor import FinancialExtractor
from backend.ratio_calculator import compute_ratios

router = APIRouter()


def _save_temp(upload: UploadFile) -> str:
    """Save an UploadFile to a temporary path and return the path."""
    suffix = os.path.splitext(upload.filename or ".pdf")[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    shutil.copyfileobj(upload.file, tmp)
    tmp.close()
    return tmp.name


def _extract_pdf(path: str) -> dict:
    """Extract data from a single PDF. Returns dict of line items."""
    extractor = FinancialExtractor()
    return extractor.extract_from_pdf(path)


@router.post("/upload")
async def upload_financials(
    statements_current: UploadFile = File(...),
    statements_previous: UploadFile = File(...),
    company_name: str = Form(default="Company"),
):
    """
    Accept 2 PDF files (Current Year, Previous Year) which contain Balance Sheet, 
    P&L, and Cash Flow data, extract data, compute all 48 ratios, and return the feature vector.
    """
    tmp_paths = []
    try:
        # Save all files to temp
        files = {
            "statements_current": statements_current,
            "statements_previous": statements_previous,
        }
        saved = {}
        for key, f in files.items():
            path = _save_temp(f)
            tmp_paths.append(path)
            saved[key] = path

        # Extract from each PDF
        current = _extract_pdf(saved["statements_current"])
        previous = _extract_pdf(saved["statements_previous"])

        items_current = len(current)
        items_previous = len(previous)

        if items_current < 3:
            raise HTTPException(
                422,
                f"Could not extract enough data from current-year PDFs. "
                f"Only {items_current} items found. "
                f"Please check that the uploaded files contain readable financial tables."
            )

        # Compute the 48 ratios
        ratios = compute_ratios(current, previous if items_previous > 0 else None)

        return JSONResponse(content={
            "company_name": company_name,
            "features": ratios,
            "items_extracted": {
                "current_year": items_current,
                "previous_year": items_previous,
            },
            "raw_data": {
                "current": {k: round(v, 2) if isinstance(v, float) else v for k, v in current.items()},
                "previous": {k: round(v, 2) if isinstance(v, float) else v for k, v in previous.items()},
            },
            "message": f"Successfully extracted {items_current} current-year and {items_previous} previous-year items."
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error processing files: {str(e)}")
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
