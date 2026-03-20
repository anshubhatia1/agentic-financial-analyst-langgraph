import re
from datetime import date
from dateutil.relativedelta import relativedelta

def parse_date_range(user_query: str) -> tuple[str, str]:
    """
    Return (start_date, end_date) as "YYYY-MM-DD" strings.

    Priority: explicit patterns are tried in order; first match wins.
    Falls back to the last 12 months if no pattern matches.
    """
    today = date.today()
    q     = user_query.lower()

    # "last N months"
    m = re.search(r"last\s+(\d+)\s+months?", q)
    if m:
        n = int(m.group(1))
        return str(today - relativedelta(months=n)), str(today)

    # "last N years"
    m = re.search(r"last\s+(\d+)\s+years?", q)
    if m:
        n = int(m.group(1))
        return str(today - relativedelta(years=n)), str(today)

    # "last N quarters"
    m = re.search(r"last\s+(\d+)\s+quarters?", q)
    if m:
        n = int(m.group(1))
        return str(today - relativedelta(months=3 * n)), str(today)

    # "last quarter" / "past quarter"
    if re.search(r"\b(last|past)\s+quarter\b", q):
        return str(today - relativedelta(months=3)), str(today)

    # "last year" / "past year"
    if re.search(r"\b(last|past)\s+year\b", q):
        return str(today - relativedelta(years=1)), str(today)

    # "year to date" / "ytd"
    if re.search(r"\bytd\b|year[\s-]to[\s-]date", q):
        return str(date(today.year, 1, 1)), str(today)

    # "Q1/Q2/Q3/Q4 YYYY"
    m = re.search(r"q([1-4])\s*(20\d{2})", q)
    if m:
        qnum, year  = int(m.group(1)), int(m.group(2))
        start_month = (qnum - 1) * 3 + 1
        end_month   = qnum * 3
        # Days in each month (ignoring leap-year edge case for Feb end)
        _days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        end_day = _days[end_month - 1]
        return f"{year}-{start_month:02d}-01", f"{year}-{end_month:02d}-{end_day}"

    # Bare year: "in 2023" / "for 2023" / "2023"
    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        year = int(m.group(1))
        return f"{year}-01-01", f"{year}-12-31"

    # Default: last 12 months
    return str(today - relativedelta(months=12)), str(today)