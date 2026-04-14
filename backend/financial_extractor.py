"""
Financial Statement Extractor
Parses PDF financial statements (Balance Sheet, P&L, Cash Flow)
and extracts raw financial line items using keyword matching.

Optimised for Indian annual reports where pdfplumber.extract_text()
produces lines like:
    (a) Property, Plant and Equipment 2 18599.88 16754.31
    Total Equity 52625.69 41852.59
"""

import re
import logging

logger = logging.getLogger(__name__)

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


class FinancialExtractor:
    """Extracts financial data from PDF documents."""

    def __init__(self):
        self.data = {}
        self.scale = 1.0

    # ── Keywords for line-item matching ─────────────────────────
    LINE_ITEMS = {
        # Balance Sheet
        "cash": [
            "cash and cash equivalents", "cash & cash equivalents",
            "cash and bank balances", "cash and bank",
        ],
        "accounts_receivable": [
            "trade receivables", "accounts receivable",
            "debtors", "sundry debtors",
        ],
        "inventory": [
            "inventories", "inventory", "stock in trade", "stocks",
        ],
        "current_assets": ["total current assets"],
        "fixed_assets": [
            "property, plant and equipment", "property plant and equipment",
            "ppe", "net fixed assets",
        ],
        "total_non_current_assets": ["total non-current assets", "total non current assets"],
        "total_assets": ["total assets"],
        "current_liabilities": ["total current liabilities"],
        "long_term_debt": [
            "long term debt", "long-term borrowings", "non-current borrowings",
            "term loans", "long term borrowings",
        ],
        "total_liabilities": ["total liabilities"],
        "retained_earnings": [
            "retained earnings", "reserves and surplus", "retained profit",
            "other equity",
        ],
        "total_equity": [
            "total equity",
        ],
        "shareholders_equity": [
            "shareholders equity", "stockholders equity",
            "net worth", "total shareholders funds", "shareholders' equity",
            "stockholders' equity", "total shareholders' funds",
        ],
        "borrowings": [
            "borrowings",
        ],
        "interest_bearing_debt": [
            "interest bearing debt", "interest-bearing debt",
            "total borrowings", "total debt",
        ],
        "paid_in_capital": [
            "equity share capital", "share capital",
            "paid-in capital", "paid in capital",
            "issued capital", "common stock",
        ],
        "contingent_liabilities": [
            "contingent liabilities", "contingent liability",
        ],

        # Profit & Loss
        "revenue": [
            "revenue from operations", "net sales", "total revenue",
            "turnover", "net revenue", "total income from operations",
        ],
        "total_income": ["total income"],
        "cost_of_goods": [
            "cost of materials consumed", "cost of goods sold",
            "cost of revenue", "cost of sales", "cogs",
        ],
        "gross_profit": ["gross profit", "gross margin"],
        "operating_expenses": [
            "total expenses",
        ],
        "operating_profit": [
            "profit from operations", "operating profit",
            "operating income", "income from operations",
        ],
        "ebit": ["ebit"],
        "ebitda": ["ebitda", "earnings before interest"],
        "interest_expense": [
            "finance costs", "interest expense", "finance charges",
            "borrowing costs", "interest and finance charges",
        ],
        "tax": [
            "total tax expense", "tax expense",
            "income tax expense", "provision for tax",
        ],
        "net_profit": [
            "profit for the year", "profit/(loss) for the year",
            "profit after tax", "net profit", "net income",
            "pat", "profit for the period",
        ],
        "profit_before_tax": [
            "profit before tax", "profit/(loss) before tax",
            "income before tax", "pbt", "profit before taxation",
        ],
        "depreciation": [
            "depreciation, depletion & amortisation",
            "depreciation, depletion and amortisation",
            "depreciation and amortization", "depreciation & amortization",
            "depreciation and amortisation", "depreciation",
        ],
        "rd_expense": [
            "research and development", "r&d expenses",
            "research and development expenses",
        ],
        "non_operating_income": [
            "other income",
        ],

        # Cash Flow
        "operating_cashflow": [
            "net cash from / (used in) operating",
            "net cash from operating", "cash from operations",
            "net cash provided by operating",
            "net cash generated from operating",
            "cash flows from operating",
        ],
        "investing_cashflow": [
            "net cash from / (used in) investing",
            "net cash from investing", "cash used in investing",
            "net cash used in investing",
        ],
        "financing_cashflow": [
            "net cash from / (used in) financing",
            "net cash from financing", "cash used in financing",
        ],
        "capex": [
            "other capital expenditure", "capital expenditure",
            "purchase of property", "purchase of fixed assets",
            "additions to fixed assets",
        ],
        "free_cashflow": ["free cash flow", "fcf"],
        "shares_outstanding": [
            "shares outstanding", "number of shares", "weighted average shares",
            "total shares", "number of equity shares",
        ],
    }

    def _parse_number(self, text: str):
        """Parse a number string, handling parentheses for negatives, commas, currency symbols."""
        if not text:
            return None
        text = str(text).strip()
        if text in ("", "-", "—", "nil", "Nil", "NIL", "n/a", "N/A"):
            return None
        negative = ("(" in text and ")" in text) or text.startswith("-")
        cleaned = re.sub(r"[₹$£€%\s]", "", text)
        cleaned = cleaned.replace("(", "").replace(")", "").replace(",", "")
        try:
            val = float(cleaned)
            if val == 0:
                return None
            return -abs(val) if negative else val
        except (ValueError, TypeError):
            return None

    def _detect_scale(self, text: str) -> float:
        t = text.lower()
        if any(x in t for x in ["in crores", "rs. crores", "crore", "₹ crore", "` in crore"]):
            return 10_000_000  # 1 Crore = 10 million
        if any(x in t for x in ["in lakhs", "rs. lakhs", "lakh", "lacs"]):
            return 100_000
        if any(x in t for x in ["in millions", "million", "$ million"]):
            return 1_000_000
        if any(x in t for x in ["in thousands", "thousands"]):
            return 1_000
        return 1.0

    def _extract_trailing_numbers(self, line: str):
        """
        From a line like '(a) Property, Plant and Equipment 2 18599.88 16754.31',
        extract the trailing numbers, skipping note reference numbers.
        Returns list of parsed floats in order.
        """
        # Find all number-like tokens in the line
        tokens = re.findall(r"\(?\-?\d[\d,]*(?:\.\d+)?\)?", line)
        if not tokens:
            return []

        results = []
        for t in tokens:
            val = self._parse_number(t)
            if val is not None:
                results.append(val)
        return results

    def _pick_current_year_value(self, numbers: list):
        """
        Given a list of numbers from a line, pick the current-year value.
        In Indian annual reports, the format is usually:
            Label NoteNo CurrentYearValue PreviousYearValue
        We want the current year value (first number with decimal point or
        the first 'large' number, skipping small integer note numbers).
        """
        if not numbers:
            return None

        # Filter: anything with a decimal is definitely a financial value
        # Small integers (1-99) without decimals are likely note numbers
        financial = []
        for n in numbers:
            abs_n = abs(n)
            # If it has decimals or is >= 100, it's likely a financial value
            if abs_n != int(abs_n) or abs_n >= 100:
                financial.append(n)

        if financial:
            return financial[0]  # First financial number = current year

        # If all numbers are small, the first one might be a note number
        # The second one is probably the current year value
        if len(numbers) >= 2:
            return numbers[1]
        return numbers[0]

    def extract_from_pdf(self, path: str) -> dict:
        """
        Extract financial data from a single PDF file.
        Returns a dict of extracted line items.
        """
        data = {}
        self.scale = 1.0
        all_text = ""

        # Pass 1: pdfplumber for text
        if pdfplumber is not None:
            try:
                with pdfplumber.open(path) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        text = page.extract_text() or ""
                        all_text += f"\n--- PAGE {page_num} ---\n{text}"
                        if self.scale == 1.0:
                            self.scale = self._detect_scale(text)
            except Exception as e:
                logger.warning(f"pdfplumber failed on {path}: {e}")

        # Pass 2: PyMuPDF fallback for text
        if not all_text.strip() and fitz is not None:
            try:
                doc = fitz.open(path)
                for page_num, page in enumerate(doc, 1):
                    t = page.get_text("text")
                    if t:
                        all_text += f"\n--- PAGE {page_num} ---\n{t}"
                        if self.scale == 1.0:
                            self.scale = self._detect_scale(t)
                doc.close()
            except Exception as e:
                logger.warning(f"PyMuPDF failed on {path}: {e}")

        if not all_text.strip():
            logger.warning(f"Could not extract any text from {path}")
            return data

        # Line-by-line extraction
        lines = all_text.split("\n")
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            # Remove leading markers like (a), (b), (i), (ii)
            line_clean = re.sub(r"^\s*\([a-z]+\)\s*", "", line_lower)
            line_clean = re.sub(r"^\s*\([ivx]+\)\s*", "", line_clean)
            line_clean = re.sub(r"^\s*[ivx]+\.\s+", "", line_clean)  # Roman numerals
            line_clean = re.sub(r"^\s*\d+\.\s+", "", line_clean)     # "1. "
            line_clean = line_clean.strip()

            if not line_clean or len(line_clean) < 3:
                continue

            for item, keywords in self.LINE_ITEMS.items():
                if item in data:
                    continue
                for kw in keywords:
                    if kw in line_clean:
                        # Get numbers from this line
                        numbers = self._extract_trailing_numbers(line)
                        val = self._pick_current_year_value(numbers)

                        # If no good number found on this line, check next line
                        # (some items wrap across lines)
                        if val is None and i + 1 < len(lines):
                            combined = line + " " + lines[i + 1]
                            numbers = self._extract_trailing_numbers(combined)
                            val = self._pick_current_year_value(numbers)

                        if val is not None:
                            data[item] = val
                            break
                if item in data:
                    break

        # Apply scale factor
        if self.scale != 1.0:
            for k in data:
                if k != "shares_outstanding":
                    data[k] = data[k] * self.scale

        logger.info(f"Extracted {len(data)} items from {path} (scale={self.scale})")
        return data
