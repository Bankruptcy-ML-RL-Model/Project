"""
Financial Ratio Calculator
Computes all 48 ratios required by the XGBoost bankruptcy prediction model
from raw extracted financial statement data.
"""


def _sd(a, b, eps=1e-8):
    """Safe division."""
    try:
        return float(a) / (float(b) + eps) if float(b) != 0 else 0.0
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0


def _growth(current, previous, eps=1e-8):
    """Compute growth rate between two periods."""
    try:
        c = float(current)
        p = float(previous)
        if abs(p) < eps:
            return 0.0
        return (c - p) / (abs(p) + eps)
    except (ValueError, TypeError):
        return 0.0


def _apply_sanity_fixes(data: dict) -> dict:
    """Fix common extraction errors in financial data."""
    d = dict(data)  # copy
    eps = 1e-8

    ta = d.get("total_assets", 0)
    te = d.get("total_equity", 0)
    tl = d.get("total_liabilities", 0)
    ca = d.get("current_assets", 0)
    rev = d.get("revenue", 0)
    op = d.get("operating_profit", 0)
    np_ = d.get("net_profit", 0)
    pbt = d.get("profit_before_tax", 0)
    tax = d.get("tax", 0)
    csh = d.get("cash", 0)
    ar = d.get("accounts_receivable", 0)

    # Tax cannot exceed profit before tax
    if pbt > 0 and tax > pbt:
        d["tax"] = pbt * 0.30

    # Revenue cannot be less than operating profit
    if op > 0 and (rev == 0 or rev < op):
        d["revenue"] = op / 0.10

    # Revenue cannot be less than net profit
    if np_ > 0 and d.get("revenue", 0) < np_:
        d["revenue"] = np_ / 0.08

    # Total liabilities cannot exceed 2x total assets
    if ta > 0 and tl > ta * 2 and te > 0:
        d["total_liabilities"] = ta - te

    # Current assets cannot exceed 2x total assets
    if ta > 0 and ca > ta * 2:
        d["current_assets"] = ta * 0.45

    # Cash too low vs assets
    if ta > 0 and 0 < csh < ta * 0.00001:
        d["cash"] = ta * 0.05

    # AR too low vs assets
    if ta > 0 and 0 < ar < ta * 0.00001:
        d["accounts_receivable"] = ta * 0.08

    return d


def compute_ratios(current: dict, previous: dict = None) -> list:
    """
    Compute all 48 financial ratios from extracted data.

    Parameters
    ----------
    current : dict
        Extracted financial data for the current year.
    previous : dict or None
        Extracted financial data for the previous year (for growth rates).

    Returns
    -------
    list of float
        48-element list matching feature_metadata.json order.
    """
    c = _apply_sanity_fixes(current)
    p = _apply_sanity_fixes(previous) if previous else {}
    eps = 1e-8

    # ── Extract current-year values ────────────────────────────
    ta = c.get("total_assets", eps)
    te = c.get("total_equity", eps)
    tl = c.get("total_liabilities", eps)
    ca = c.get("current_assets", 0)
    cl = c.get("current_liabilities", eps)
    inv = c.get("inventory", 0)
    fa = c.get("fixed_assets", 0)
    ltd = c.get("long_term_debt", 0)
    re_ = c.get("retained_earnings", 0)
    csh = c.get("cash", 0)
    ar = c.get("accounts_receivable", eps)
    ibd = c.get("interest_bearing_debt", ltd)  # fallback to LTD
    pic = c.get("paid_in_capital", te)  # fallback to equity
    cont_liab = c.get("contingent_liabilities", 0)

    rev = c.get("revenue", eps)
    cogs = c.get("cost_of_goods", 0)
    gross = c.get("gross_profit", rev - cogs if rev > eps and cogs > 0 else 0)
    op = c.get("operating_profit", 0)
    ebitda = c.get("ebitda", 0)
    ie = c.get("interest_expense", eps)
    tax = c.get("tax", 0)
    np_ = c.get("net_profit", 0)
    pbt = c.get("profit_before_tax", np_ + tax)
    dep = c.get("depreciation", 0)
    rd = c.get("rd_expense", 0)

    op_cf = c.get("operating_cashflow", np_ + dep)
    capex = c.get("capex", 0)
    shares = c.get("shares_outstanding", 0)

    wc = ca - cl

    # If EBIT not explicitly extracted, estimate
    if op == 0 and ebitda > 0:
        op = ebitda - dep

    # ── Previous-year values for growth rates ─────────────────
    p_gross = p.get("gross_profit", p.get("revenue", 0) - p.get("cost_of_goods", 0))
    p_op = p.get("operating_profit", 0)
    p_np = p.get("net_profit", 0)
    p_ta = p.get("total_assets", 0)
    p_te = p.get("total_equity", 0)
    p_rev = p.get("revenue", 0)
    p_roa = _sd(p_np, p_ta) if p_ta > 0 else 0

    # ── Per-share values ──────────────────────────────────────
    if shares and shares > 0:
        nvps = _sd(te, shares)
        eps_val = _sd(np_, shares)
        cfps = _sd(op_cf, shares)
        rvps = _sd(rev, shares)
        opps = _sd(op, shares)
        pbtps = _sd(pbt, shares)
    else:
        # Proxy: use ratio to total assets
        nvps = _sd(te, ta)
        eps_val = _sd(np_, rev)
        cfps = _sd(op_cf, rev)
        rvps = _sd(rev, ta)
        opps = _sd(op, ta)
        pbtps = _sd(pbt, ta)

    # ── Compute all 48 ratios in exact metadata order ─────────
    ratios = [
        # 1. ROA(C) before interest and depreciation before interest
        _sd(op + dep, ta),
        # 2. ROA(A) before interest and % after tax
        _sd(np_, ta),
        # 3. ROA(B) before interest and depreciation after tax
        _sd(op, ta),
        # 4. Operating Gross Margin
        _sd(gross, rev),
        # 5. Realized Sales Gross Margin
        _sd(gross, rev),
        # 6. Operating Profit Rate
        _sd(op, rev),
        # 7. Pre-tax net Interest Rate
        _sd(op + ie, rev),
        # 8. After-tax net Interest Rate
        _sd(np_, rev),
        # 9. Non-industry income and expenditure/revenue
        _sd(c.get("non_operating_income", 0), rev),
        # 10. Continuous interest rate (after tax)
        _sd(ie, rev),
        # 11. Operating Expense Rate
        _sd(rev - op, rev),
        # 12. Research and development expense rate
        _sd(rd, rev),
        # 13. Cash flow rate
        _sd(op_cf, rev),
        # 14. Interest-bearing debt interest rate
        _sd(ie, ibd + eps),
        # 15. Tax rate (A)
        _sd(tax, pbt + eps),
        # 16. Net Value Per Share (B)
        nvps,
        # 17. Net Value Per Share (A)
        nvps,
        # 18. Net Value Per Share (C)
        nvps,
        # 19. Persistent EPS in the Last Four Seasons
        eps_val,
        # 20. Cash Flow Per Share
        cfps,
        # 21. Revenue Per Share (Yuan ¥)
        rvps,
        # 22. Operating Profit Per Share (Yuan ¥)
        opps,
        # 23. Per Share Net profit before tax (Yuan ¥)
        pbtps,
        # 24. Realized Sales Gross Profit Growth Rate
        _growth(gross, p_gross) if p else 0.0,
        # 25. Operating Profit Growth Rate
        _growth(op, p_op) if p else 0.0,
        # 26. After-tax Net Profit Growth Rate
        _growth(np_, p_np) if p else 0.0,
        # 27. Regular Net Profit Growth Rate
        _growth(np_, p_np) if p else 0.0,
        # 28. Continuous Net Profit Growth Rate
        _growth(np_, p_np) if p else 0.0,
        # 29. Total Asset Growth Rate
        _growth(ta, p_ta) if p else 0.0,
        # 30. Net Value Growth Rate
        _growth(te, p_te) if p else 0.0,
        # 31. Total Asset Return Growth Rate Ratio
        _growth(_sd(np_, ta), p_roa) if p else 0.0,
        # 32. Cash Reinvestment %
        _sd(op_cf, ta),
        # 33. Current Ratio
        _sd(ca, cl),
        # 34. Quick Ratio
        _sd(ca - inv, cl),
        # 35. Interest Expense Ratio
        _sd(ie, op + ie),
        # 36. Total debt/Total net worth
        _sd(tl, te),
        # 37. Debt ratio %
        _sd(tl, ta),
        # 38. Net worth/Assets
        _sd(te, ta),
        # 39. Long-term fund suitability ratio (A)
        _sd(te + ltd, fa + eps),
        # 40. Borrowing dependency
        _sd(ltd, ta),
        # 41. Contingent liabilities/Net worth
        _sd(cont_liab, te),
        # 42. Operating profit/Paid-in capital
        _sd(op, pic),
        # 43. Net Income to Stockholder's Equity
        _sd(np_, te),
        # 44. Liability to Equity
        _sd(tl, te),
        # 45. Degree of Financial Leverage (DFL)
        _sd(op, op - ie + eps),
        # 46. Interest Coverage Ratio (Interest expense to EBIT)
        _sd(op, ie),
        # 47. Net Income Flag
        1.0 if np_ > 0 else 0.0,
        # 48. Equity to Liability
        _sd(te, tl),
    ]

    return ratios
