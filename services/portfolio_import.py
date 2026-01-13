"""Portfolio import service - Parse CSV files from various brokerages."""

import csv
import io
import base64
from dataclasses import dataclass
from typing import Optional

from data.etf_universe import get_universe_dict


@dataclass
class ImportedHolding:
    """A single imported holding."""
    ticker: str
    shares: float
    value: Optional[float] = None
    name: Optional[str] = None
    price: Optional[float] = None


@dataclass
class ImportResult:
    """Result of portfolio import."""
    success: bool
    holdings: list[ImportedHolding]
    total_value: float
    matched_etfs: int
    unmatched_tickers: list[str]
    warnings: list[str]
    error: Optional[str] = None


# Common column name mappings for different brokerages
TICKER_COLUMNS = [
    "symbol", "ticker", "sym", "stock symbol", "security",
    "investment", "fund", "etf", "holding", "name"
]

SHARES_COLUMNS = [
    "shares", "quantity", "qty", "units", "share balance",
    "total shares", "number of shares", "holdings"
]

VALUE_COLUMNS = [
    "value", "market value", "current value", "total value",
    "amount", "balance", "position value", "market_value"
]

PRICE_COLUMNS = [
    "price", "last price", "current price", "closing price",
    "last", "quote", "nav"
]

NAME_COLUMNS = [
    "name", "description", "security name", "fund name",
    "security description", "investment name"
]


def _normalize_column(col: str) -> str:
    """Normalize column name for matching."""
    return col.lower().strip().replace("_", " ").replace("-", " ")


def _find_column(headers: list[str], candidates: list[str]) -> Optional[int]:
    """Find the index of a column matching one of the candidates."""
    normalized_headers = [_normalize_column(h) for h in headers]
    for candidate in candidates:
        if candidate in normalized_headers:
            return normalized_headers.index(candidate)
    return None


def _parse_number(value: str) -> Optional[float]:
    """Parse a numeric value, handling currency symbols and commas."""
    if not value:
        return None
    cleaned = value.replace("$", "").replace(",", "").replace(" ", "").strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]
    try:
        return float(cleaned)
    except ValueError:
        return None


def _clean_ticker(ticker: str) -> str:
    """Clean and normalize ticker symbol."""
    cleaned = ticker.upper().strip()
    for suffix in [" CLASS A", " CLASS B", " CL A", " CL B", "-A", "-B"]:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)]
    cleaned = cleaned.replace("*", "").replace("^", "").strip()
    return cleaned


def parse_csv_content(content: str) -> ImportResult:
    """Parse CSV content and extract holdings."""
    warnings = []
    holdings = []
    unmatched = []
    
    try:
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        
        if len(rows) < 2:
            return ImportResult(
                success=False, holdings=[], total_value=0,
                matched_etfs=0, unmatched_tickers=[], warnings=[],
                error="CSV file must have at least a header row and one data row"
            )
        
        header_idx = 0
        for i, row in enumerate(rows):
            if any(cell.strip() for cell in row):
                header_idx = i
                break
        
        headers = rows[header_idx]
        data_rows = rows[header_idx + 1:]
        
        ticker_col = _find_column(headers, TICKER_COLUMNS)
        shares_col = _find_column(headers, SHARES_COLUMNS)
        value_col = _find_column(headers, VALUE_COLUMNS)
        price_col = _find_column(headers, PRICE_COLUMNS)
        name_col = _find_column(headers, NAME_COLUMNS)
        
        if ticker_col is None:
            return ImportResult(
                success=False, holdings=[], total_value=0,
                matched_etfs=0, unmatched_tickers=[], warnings=[],
                error=f"Could not find ticker/symbol column. Found: {headers}"
            )
        
        if shares_col is None and value_col is None:
            return ImportResult(
                success=False, holdings=[], total_value=0,
                matched_etfs=0, unmatched_tickers=[], warnings=[],
                error="Could not find shares or value column"
            )
        
        etf_universe = get_universe_dict()
        
        for row_num, row in enumerate(data_rows, start=header_idx + 2):
            if not any(cell.strip() for cell in row):
                continue
            if len(row) <= ticker_col:
                continue
            
            raw_ticker = row[ticker_col].strip()
            if not raw_ticker:
                continue
            
            skip_keywords = ["cash", "pending", "money market", "sweep", "core", 
                           "total", "subtotal", "account"]
            if any(kw in raw_ticker.lower() for kw in skip_keywords):
                continue
            
            ticker = _clean_ticker(raw_ticker)
            
            shares = None
            if shares_col is not None and len(row) > shares_col:
                shares = _parse_number(row[shares_col])
            
            value = None
            if value_col is not None and len(row) > value_col:
                value = _parse_number(row[value_col])
            
            price = None
            if price_col is not None and len(row) > price_col:
                price = _parse_number(row[price_col])
            
            name = None
            if name_col is not None and len(row) > name_col:
                name = row[name_col].strip()
            
            if shares is None and value is not None and price is not None and price > 0:
                shares = value / price
            elif value is None and shares is not None and price is not None:
                value = shares * price
            
            if shares is None and value is None:
                warnings.append(f"Row {row_num}: Could not determine position for {ticker}")
                continue
            
            if ticker not in etf_universe:
                unmatched.append(ticker)
            
            holdings.append(ImportedHolding(
                ticker=ticker, shares=shares or 0,
                value=value, name=name, price=price,
            ))
        
        if not holdings:
            return ImportResult(
                success=False, holdings=[], total_value=0,
                matched_etfs=0, unmatched_tickers=[], warnings=warnings,
                error="No valid holdings found in CSV"
            )
        
        total_value = sum(h.value for h in holdings if h.value is not None)
        
        if total_value == 0 and all(h.shares > 0 for h in holdings):
            warnings.append("No market values found. Values will be fetched.")
        
        matched = len([h for h in holdings if h.ticker in etf_universe])
        
        if unmatched:
            warnings.append(f"Tickers not in ETF universe: {', '.join(unmatched[:10])}")
        
        return ImportResult(
            success=True, holdings=holdings, total_value=total_value,
            matched_etfs=matched, unmatched_tickers=unmatched, warnings=warnings,
        )
        
    except Exception as e:
        return ImportResult(
            success=False, holdings=[], total_value=0,
            matched_etfs=0, unmatched_tickers=[], warnings=[],
            error=f"Error parsing CSV: {str(e)}"
        )


def parse_uploaded_file(contents: str, filename: str) -> ImportResult:
    """Parse an uploaded file from Dash upload component."""
    if not contents:
        return ImportResult(
            success=False, holdings=[], total_value=0,
            matched_etfs=0, unmatched_tickers=[], warnings=[],
            error="No file content provided"
        )
    
    if not filename.lower().endswith(".csv"):
        return ImportResult(
            success=False, holdings=[], total_value=0,
            matched_etfs=0, unmatched_tickers=[], warnings=[],
            error="Only CSV files are supported"
        )
    
    try:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string).decode("utf-8")
        return parse_csv_content(decoded)
    except Exception as e:
        return ImportResult(
            success=False, holdings=[], total_value=0,
            matched_etfs=0, unmatched_tickers=[], warnings=[],
            error=f"Error reading file: {str(e)}"
        )


def convert_to_portfolio_weights(
    holdings: list[ImportedHolding],
    fetch_missing_prices: bool = True
) -> tuple[list[dict], float, list[str]]:
    """Convert imported holdings to portfolio weights."""
    warnings = []
    
    if fetch_missing_prices and any(h.value is None for h in holdings):
        try:
            import yfinance as yf
            
            tickers_to_fetch = [h.ticker for h in holdings if h.value is None]
            if tickers_to_fetch:
                data = yf.download(tickers_to_fetch, period="1d", progress=False)
                if "Close" in data.columns:
                    prices = data["Close"].iloc[-1]
                    if len(tickers_to_fetch) == 1:
                        prices = {tickers_to_fetch[0]: prices}
                    else:
                        prices = prices.to_dict()
                    
                    for h in holdings:
                        if h.value is None and h.ticker in prices:
                            price = prices[h.ticker]
                            if price and price > 0:
                                h.value = h.shares * price
                                h.price = price
        except Exception as e:
            warnings.append(f"Could not fetch prices: {str(e)}")
    
    total_value = sum(h.value for h in holdings if h.value is not None and h.value > 0)
    
    if total_value <= 0:
        return [], 0, ["Could not calculate portfolio value"]
    
    result = []
    for h in holdings:
        if h.value is not None and h.value > 0:
            weight = h.value / total_value
            rationale = f"Imported: {h.shares:.2f} shares"
            if h.price:
                rationale += f" @ ${h.price:.2f}"
            result.append({
                "ticker": h.ticker,
                "weight": weight,
                "shares": h.shares,
                "value": h.value,
                "rationale": rationale,
            })
    
    result.sort(key=lambda x: x["weight"], reverse=True)
    return result, total_value, warnings


def generate_sample_csv() -> str:
    """Generate a sample CSV for users to reference."""
    return """Symbol,Shares,Price,Value
SPY,100,450.00,45000.00
VTI,50,220.00,11000.00
AGG,200,100.00,20000.00
VWO,150,42.00,6300.00
GLD,30,180.00,5400.00
"""
