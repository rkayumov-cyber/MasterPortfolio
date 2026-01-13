"""Predefined stress test scenarios."""

from datetime import date

from domain.schemas import StressScenario

# Historical stress periods
HISTORICAL_PERIODS = {
    "GFC 2008-2009": {
        "name": "Global Financial Crisis",
        "start": date(2008, 9, 1),
        "end": date(2009, 3, 31),
        "description": "Lehman collapse and financial crisis",
    },
    "COVID Crash 2020": {
        "name": "COVID-19 Market Crash",
        "start": date(2020, 2, 19),
        "end": date(2020, 3, 23),
        "description": "Pandemic-induced market selloff",
    },
    "2022 Rate Shock": {
        "name": "2022 Fed Tightening",
        "start": date(2022, 1, 1),
        "end": date(2022, 10, 31),
        "description": "Aggressive Fed rate hikes, inflation concerns",
    },
    "Dot-com Bust": {
        "name": "Dot-com Bubble Burst",
        "start": date(2000, 3, 1),
        "end": date(2002, 10, 31),
        "description": "Technology bubble burst",
    },
    "Euro Crisis 2011": {
        "name": "European Debt Crisis",
        "start": date(2011, 7, 1),
        "end": date(2011, 10, 31),
        "description": "European sovereign debt concerns",
    },
}

# Hypothetical shock scenarios
# Values represent percentage shocks to each asset class
HYPOTHETICAL_SCENARIOS: list[StressScenario] = [
    StressScenario(
        name="Equity Selloff",
        description="15% decline in global equities, flight to quality",
        shocks={
            "Equity": -0.15,
            "Fixed Income": 0.03,
            "Alternatives": -0.05,
            "Inverse": 0.15,
        },
    ),
    StressScenario(
        name="Rate Shock +200bp",
        description="Sharp rise in interest rates",
        shocks={
            "Equity": -0.05,
            "Fixed Income": -0.10,  # Bonds hurt by rates
            "Alternatives": 0.02,
            "Inverse": 0.05,
        },
    ),
    StressScenario(
        name="Tech Crash",
        description="Technology sector drops 25%, broader market -10%",
        shocks={
            "Equity": -0.10,
            "Fixed Income": 0.02,
            "Alternatives": -0.03,
            "Inverse": 0.10,
        },
    ),
    StressScenario(
        name="Stagflation",
        description="High inflation with slowing growth",
        shocks={
            "Equity": -0.12,
            "Fixed Income": -0.08,
            "Alternatives": 0.10,  # Gold benefits
            "Inverse": 0.12,
        },
    ),
    StressScenario(
        name="Risk-On Rally",
        description="Strong economic recovery, equities rally",
        shocks={
            "Equity": 0.15,
            "Fixed Income": -0.03,
            "Alternatives": 0.05,
            "Inverse": -0.15,
        },
    ),
    StressScenario(
        name="USD Weakness",
        description="Dollar weakens, international assets benefit",
        shocks={
            "Equity": 0.02,
            "Fixed Income": -0.01,
            "Alternatives": 0.08,
            "Inverse": -0.02,
        },
    ),
]

# Sector-specific shocks (for sector-tilted portfolios)
SECTOR_SHOCKS = {
    "Technology": {
        "tech_crash": -0.25,
        "tech_boom": 0.20,
    },
    "Financials": {
        "credit_crisis": -0.30,
        "rate_normalization": 0.15,
    },
    "Energy": {
        "oil_crash": -0.35,
        "oil_spike": 0.25,
    },
    "Healthcare": {
        "regulatory_risk": -0.15,
        "defensive_bid": 0.10,
    },
    "Real Estate": {
        "rate_shock": -0.20,
        "rate_cut": 0.15,
    },
}


def get_historical_periods() -> dict:
    """Return historical stress periods."""
    return HISTORICAL_PERIODS.copy()


def get_hypothetical_scenarios() -> list[StressScenario]:
    """Return hypothetical stress scenarios."""
    return HYPOTHETICAL_SCENARIOS.copy()


def get_scenario_by_name(name: str) -> StressScenario | None:
    """Get a specific scenario by name."""
    for scenario in HYPOTHETICAL_SCENARIOS:
        if scenario.name == name:
            return scenario
    return None
