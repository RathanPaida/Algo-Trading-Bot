"""The stock universe: NIFTY 50 constituents (2025 list) and their sectors."""

SECTOR_MAP: dict[str, str] = {
    "ADANIENT.NS": "Infrastructure", "ADANIPORTS.NS": "Infrastructure",
    "APOLLOHOSP.NS": "Healthcare", "ASIANPAINT.NS": "Consumer",
    "AXISBANK.NS": "Banking", "BAJAJ-AUTO.NS": "Auto",
    "BAJFINANCE.NS": "NBFC", "BAJAJFINSV.NS": "NBFC",
    "BHARTIARTL.NS": "Telecom", "BPCL.NS": "Energy",
    "BRITANNIA.NS": "Consumer", "CIPLA.NS": "Pharma",
    "COALINDIA.NS": "Energy", "DIVISLAB.NS": "Pharma",
    "DRREDDY.NS": "Pharma", "EICHERMOT.NS": "Auto",
    "GRASIM.NS": "Diversified", "HCLTECH.NS": "IT",
    "HDFCBANK.NS": "Banking", "HDFCLIFE.NS": "Insurance",
    "HEROMOTOCO.NS": "Auto", "HINDALCO.NS": "Metals",
    "HINDUNILVR.NS": "Consumer", "ICICIBANK.NS": "Banking",
    "INDUSINDBK.NS": "Banking", "INFY.NS": "IT", "ITC.NS": "Consumer",
    "JSWSTEEL.NS": "Metals", "KOTAKBANK.NS": "Banking",
    "LT.NS": "Infrastructure", "LTIM.NS": "IT",
    "M&M.NS": "Auto", "MARUTI.NS": "Auto",
    "NESTLEIND.NS": "Consumer", "NTPC.NS": "Energy",
    "ONGC.NS": "Energy", "POWERGRID.NS": "Energy",
    "RELIANCE.NS": "Energy", "SBILIFE.NS": "Insurance",
    "SBIN.NS": "Banking", "SHRIRAMFIN.NS": "NBFC",
    "SUNPHARMA.NS": "Pharma", "TATACONSUM.NS": "Consumer",
    "TATAMOTORS.NS": "Auto", "TATASTEEL.NS": "Metals",
    "TCS.NS": "IT", "TECHM.NS": "IT",
    "TITAN.NS": "Consumer", "TRENT.NS": "Consumer",
    "ULTRACEMCO.NS": "Cement", "WIPRO.NS": "IT",
}

SYMBOLS: list[str] = list(SECTOR_MAP)

# Older / alternative Yahoo tickers to try when the main one fails.
ALIASES: dict[str, list[str]] = {
    "SHRIRAMFIN.NS": ["SRTRANSFIN.NS"],
    "TATACONSUM.NS": ["TATAGLOBAL.NS"],
    "TATAMOTORS.NS": ["TMPV.NS"],
    "LTIM.NS": ["LTM.NS"],          # LTIMindtree renamed on Yahoo
}


def get_sector(symbol: str) -> str:
    return SECTOR_MAP.get(symbol, "Unknown")
