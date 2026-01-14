"""PDF processor for extracting text from research documents."""

import base64
import re
from typing import Optional

from domain.schemas import ResearchSummary

# Try to import PyMuPDF (fitz)
try:
    import fitz

    HAS_PDF_SUPPORT = True
except ImportError:
    HAS_PDF_SUPPORT = False
    fitz = None


# Keywords for sentiment analysis
BULLISH_KEYWORDS = [
    "bullish",
    "buy",
    "overweight",
    "upgrade",
    "outperform",
    "strong buy",
    "positive",
    "upside",
    "growth",
    "rally",
    "breakout",
    "accumulate",
    "attractive",
    "opportunity",
    "momentum",
]

BEARISH_KEYWORDS = [
    "bearish",
    "sell",
    "underweight",
    "downgrade",
    "underperform",
    "reduce",
    "negative",
    "downside",
    "decline",
    "correction",
    "risk",
    "cautious",
    "avoid",
    "weakness",
    "headwinds",
]

NEUTRAL_KEYWORDS = [
    "hold",
    "neutral",
    "market perform",
    "equal weight",
    "mixed",
    "balanced",
    "uncertain",
]


def is_pdf_support_available() -> bool:
    """Check if PDF support is available."""
    return HAS_PDF_SUPPORT


def extract_text_from_base64(base64_content: str) -> Optional[str]:
    """
    Extract text from a base64-encoded PDF.

    Args:
        base64_content: Base64 string (may include data URI prefix)

    Returns:
        Extracted text or None on error
    """
    if not HAS_PDF_SUPPORT:
        return None

    try:
        # Remove data URI prefix if present
        if "," in base64_content:
            base64_content = base64_content.split(",")[1]

        # Decode base64
        pdf_bytes = base64.b64decode(base64_content)

        # Open PDF with PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        # Extract text from all pages
        text_parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_parts.append(page.get_text())

        doc.close()

        full_text = "\n".join(text_parts)
        return full_text.strip()

    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return None


def extract_text_from_bytes(pdf_bytes: bytes) -> Optional[str]:
    """
    Extract text from PDF bytes.

    Args:
        pdf_bytes: Raw PDF bytes

    Returns:
        Extracted text or None on error
    """
    if not HAS_PDF_SUPPORT:
        return None

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        text_parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_parts.append(page.get_text())

        doc.close()

        full_text = "\n".join(text_parts)
        return full_text.strip()

    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return None


def analyze_sentiment(text: str) -> str:
    """
    Analyze sentiment of text based on keyword matching.

    Returns 'bullish', 'bearish', or 'neutral'.
    """
    text_lower = text.lower()

    bullish_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
    bearish_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)
    neutral_count = sum(1 for kw in NEUTRAL_KEYWORDS if kw in text_lower)

    # Weight the counts
    bullish_score = bullish_count
    bearish_score = bearish_count
    neutral_score = neutral_count * 0.5  # Less weight on neutral

    if bullish_score > bearish_score and bullish_score > neutral_score:
        return "bullish"
    elif bearish_score > bullish_score and bearish_score > neutral_score:
        return "bearish"
    else:
        return "neutral"


def extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    """
    Extract relevant financial keywords from text.

    Returns list of found keywords.
    """
    text_lower = text.lower()
    found_keywords = []

    all_keywords = BULLISH_KEYWORDS + BEARISH_KEYWORDS + NEUTRAL_KEYWORDS

    for keyword in all_keywords:
        if keyword in text_lower and keyword not in found_keywords:
            found_keywords.append(keyword)
            if len(found_keywords) >= max_keywords:
                break

    return found_keywords


def create_preview(text: str, max_length: int = 500) -> str:
    """Create a preview of the text."""
    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= max_length:
        return text

    # Try to cut at sentence boundary
    preview = text[:max_length]
    last_period = preview.rfind(".")
    if last_period > max_length * 0.5:
        preview = preview[: last_period + 1]
    else:
        preview = preview + "..."

    return preview


def process_research_pdf(
    content: str, filename: str = "research.pdf"
) -> Optional[ResearchSummary]:
    """
    Process an uploaded research PDF and extract summary.

    Args:
        content: Base64-encoded PDF content
        filename: Original filename

    Returns:
        ResearchSummary with extracted information
    """
    # Extract text
    text = extract_text_from_base64(content)

    if text is None or len(text) < 50:
        return ResearchSummary(
            filename=filename,
            sentiment="neutral",
            word_count=0,
            preview="Unable to extract text from PDF. Please ensure the PDF contains readable text.",
            keywords=[],
        )

    # Analyze sentiment
    sentiment = analyze_sentiment(text)

    # Extract keywords
    keywords = extract_keywords(text)

    # Create preview
    preview = create_preview(text)

    # Count words
    word_count = len(text.split())

    return ResearchSummary(
        filename=filename,
        sentiment=sentiment,
        word_count=word_count,
        preview=preview,
        keywords=keywords,
    )


def get_sentiment_score(sentiment: str) -> float:
    """
    Convert sentiment label to numeric score.

    Returns -1 (bearish), 0 (neutral), or +1 (bullish).
    """
    if sentiment == "bullish":
        return 0.7
    elif sentiment == "bearish":
        return -0.7
    else:
        return 0.0
