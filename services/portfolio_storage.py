"""Portfolio Storage Service - Save and load portfolios to/from disk."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from domain.schemas import Portfolio, PortfolioHolding


# Default storage location
STORAGE_DIR = Path("saved_portfolios")
PORTFOLIO_FILE = STORAGE_DIR / "portfolios.json"


def _ensure_storage_dir():
    """Ensure the storage directory exists."""
    STORAGE_DIR.mkdir(exist_ok=True)


def _load_all_portfolios() -> dict:
    """Load all portfolios from disk."""
    _ensure_storage_dir()

    if not PORTFOLIO_FILE.exists():
        return {}

    try:
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_all_portfolios(portfolios: dict):
    """Save all portfolios to disk."""
    _ensure_storage_dir()

    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolios, f, indent=2, default=str)


def save_portfolio(
    name: str,
    holdings: list[dict],
    notes: list[str] = None,
    strategy: str = None,
    metadata: dict = None,
) -> dict:
    """
    Save a portfolio with the given name.

    Args:
        name: Portfolio name (must be unique)
        holdings: List of holding dicts with ticker, weight, rationale
        notes: Optional notes about the portfolio
        strategy: Strategy used to generate the portfolio
        metadata: Additional metadata to store

    Returns:
        dict with success status and message
    """
    if not name or not name.strip():
        return {"success": False, "message": "Portfolio name is required"}

    name = name.strip()

    if not holdings:
        return {"success": False, "message": "Portfolio has no holdings"}

    portfolios = _load_all_portfolios()

    # Create portfolio record
    portfolio_data = {
        "name": name,
        "holdings": holdings,
        "notes": notes or [],
        "strategy": strategy,
        "metadata": metadata or {},
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    # Check if updating existing
    is_update = name in portfolios
    if is_update:
        portfolio_data["created_at"] = portfolios[name].get(
            "created_at", datetime.now().isoformat()
        )

    portfolios[name] = portfolio_data
    _save_all_portfolios(portfolios)

    action = "updated" if is_update else "saved"
    return {"success": True, "message": f"Portfolio '{name}' {action} successfully"}


def load_portfolio(name: str) -> Optional[dict]:
    """
    Load a portfolio by name.

    Returns:
        Portfolio data dict or None if not found
    """
    portfolios = _load_all_portfolios()
    return portfolios.get(name)


def delete_portfolio(name: str) -> dict:
    """
    Delete a portfolio by name.

    Returns:
        dict with success status and message
    """
    portfolios = _load_all_portfolios()

    if name not in portfolios:
        return {"success": False, "message": f"Portfolio '{name}' not found"}

    del portfolios[name]
    _save_all_portfolios(portfolios)

    return {"success": True, "message": f"Portfolio '{name}' deleted"}


def list_portfolios() -> list[dict]:
    """
    List all saved portfolios.

    Returns:
        List of portfolio summaries (name, holdings count, created_at, strategy)
    """
    portfolios = _load_all_portfolios()

    summaries = []
    for name, data in portfolios.items():
        summaries.append({
            "name": name,
            "holdings_count": len(data.get("holdings", [])),
            "strategy": data.get("strategy", "Unknown"),
            "created_at": data.get("created_at", ""),
            "updated_at": data.get("updated_at", ""),
        })

    # Sort by updated_at descending (most recent first)
    summaries.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

    return summaries


def get_portfolio_names() -> list[str]:
    """Get list of all portfolio names."""
    portfolios = _load_all_portfolios()
    return sorted(portfolios.keys())


def export_portfolio_to_json(name: str) -> Optional[str]:
    """Export a portfolio as a JSON string."""
    portfolio = load_portfolio(name)
    if portfolio:
        return json.dumps(portfolio, indent=2, default=str)
    return None


def import_portfolio_from_json(json_string: str, name: str = None) -> dict:
    """
    Import a portfolio from a JSON string.

    Args:
        json_string: JSON string containing portfolio data
        name: Optional name override (uses name from JSON if not provided)

    Returns:
        dict with success status and message
    """
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        return {"success": False, "message": f"Invalid JSON: {e}"}

    # Extract required fields
    portfolio_name = name or data.get("name")
    if not portfolio_name:
        return {"success": False, "message": "Portfolio name not found in JSON"}

    holdings = data.get("holdings", [])
    if not holdings:
        return {"success": False, "message": "No holdings found in JSON"}

    return save_portfolio(
        name=portfolio_name,
        holdings=holdings,
        notes=data.get("notes", []),
        strategy=data.get("strategy"),
        metadata=data.get("metadata", {}),
    )


def rename_portfolio(old_name: str, new_name: str) -> dict:
    """
    Rename a portfolio.

    Returns:
        dict with success status and message
    """
    if not new_name or not new_name.strip():
        return {"success": False, "message": "New name is required"}

    new_name = new_name.strip()

    portfolios = _load_all_portfolios()

    if old_name not in portfolios:
        return {"success": False, "message": f"Portfolio '{old_name}' not found"}

    if new_name in portfolios and new_name != old_name:
        return {"success": False, "message": f"Portfolio '{new_name}' already exists"}

    # Get old data and update
    data = portfolios[old_name]
    data["name"] = new_name
    data["updated_at"] = datetime.now().isoformat()

    # Remove old, add new
    del portfolios[old_name]
    portfolios[new_name] = data

    _save_all_portfolios(portfolios)

    return {"success": True, "message": f"Portfolio renamed to '{new_name}'"}


def duplicate_portfolio(name: str, new_name: str) -> dict:
    """
    Duplicate a portfolio with a new name.

    Returns:
        dict with success status and message
    """
    portfolio = load_portfolio(name)
    if not portfolio:
        return {"success": False, "message": f"Portfolio '{name}' not found"}

    return save_portfolio(
        name=new_name,
        holdings=portfolio.get("holdings", []),
        notes=portfolio.get("notes", []) + [f"Duplicated from '{name}'"],
        strategy=portfolio.get("strategy"),
        metadata=portfolio.get("metadata", {}),
    )
