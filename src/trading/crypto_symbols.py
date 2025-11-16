#!/usr/bin/env python3
"""
🪙 Top 100 Cryptocurrency Symbols
قائمة أفضل 100 عملة رقمية للتداول
"""

# Top 100 cryptocurrencies by market cap (USDT pairs)
# Updated for Binance/yfinance compatibility
TOP_100_CRYPTO_SYMBOLS = [
    # Top 10 - Giants
    "BTC/USDT",   # Bitcoin
    "ETH/USDT",   # Ethereum
    "BNB/USDT",   # Binance Coin
    "XRP/USDT",   # Ripple
    "SOL/USDT",   # Solana
    "ADA/USDT",   # Cardano
    "DOGE/USDT",  # Dogecoin
    "TRX/USDT",   # Tron
    "AVAX/USDT",  # Avalanche
    "DOT/USDT",   # Polkadot

    # 11-30 - Major Alts
    "MATIC/USDT", # Polygon
    "LINK/USDT",  # Chainlink
    "LTC/USDT",   # Litecoin
    "UNI/USDT",   # Uniswap
    "ATOM/USDT",  # Cosmos
    "XLM/USDT",   # Stellar
    "ETC/USDT",   # Ethereum Classic
    "BCH/USDT",   # Bitcoin Cash
    "NEAR/USDT",  # Near Protocol
    "APT/USDT",   # Aptos
    "ALGO/USDT",  # Algorand
    "FIL/USDT",   # Filecoin
    "VET/USDT",   # VeChain
    "ICP/USDT",   # Internet Computer
    "ARB/USDT",   # Arbitrum
    "OP/USDT",    # Optimism
    "GRT/USDT",   # The Graph
    "SAND/USDT",  # The Sandbox
    "MANA/USDT",  # Decentraland
    "AXS/USDT",   # Axie Infinity

    # 31-60 - Mid-Cap
    "EOS/USDT",   # EOS
    "AAVE/USDT",  # Aave
    "FTM/USDT",   # Fantom
    "THETA/USDT", # Theta
    "EGLD/USDT",  # MultiversX
    "XTZ/USDT",   # Tezos
    "RUNE/USDT",  # THORChain
    "INJ/USDT",   # Injective
    "SNX/USDT",   # Synthetix
    "ZEC/USDT",   # Zcash
    "MKR/USDT",   # Maker
    "ENJ/USDT",   # Enjin
    "CHZ/USDT",   # Chiliz
    "BAT/USDT",   # Basic Attention Token
    "FLOW/USDT",  # Flow
    "ZIL/USDT",   # Zilliqa
    "QTUM/USDT",  # Qtum
    "ONE/USDT",   # Harmony
    "HBAR/USDT",  # Hedera
    "KAVA/USDT",  # Kava
    "CELO/USDT",  # Celo
    "COMP/USDT",  # Compound
    "KSM/USDT",   # Kusama
    "ZRX/USDT",   # 0x
    "SUSHI/USDT", # SushiSwap
    "YFI/USDT",   # yearn.finance
    "BAL/USDT",   # Balancer
    "CRV/USDT",   # Curve
    "1INCH/USDT", # 1inch
    "ANKR/USDT",  # Ankr

    # 61-80 - Emerging
    "LRC/USDT",   # Loopring
    "IMX/USDT",   # Immutable X
    "GAL/USDT",   # Galxe
    "APE/USDT",   # ApeCoin
    "GMT/USDT",   # STEPN
    "ROSE/USDT",  # Oasis Network
    "STX/USDT",   # Stacks
    "KLAY/USDT",  # Klaytn
    "AR/USDT",    # Arweave
    "DYDX/USDT",  # dYdX
    "JST/USDT",   # JUST
    "SXP/USDT",   # Solar
    "OCEAN/USDT", # Ocean Protocol
    "REN/USDT",   # Ren
    "ICX/USDT",   # ICON
    "ONT/USDT",   # Ontology
    "SC/USDT",    # Siacoin
    "ZEN/USDT",   # Horizen
    "STORJ/USDT", # Storj
    "SKL/USDT",   # SKALE

    # 81-100 - Growing Projects
    "AUDIO/USDT", # Audius
    "C98/USDT",   # Coin98
    "DUSK/USDT",  # Dusk Network
    "PEOPLE/USDT",# ConstitutionDAO
    "ALICE/USDT", # My Neighbor Alice
    "TLM/USDT",   # Alien Worlds
    "SFP/USDT",   # SafePal
    "PYR/USDT",   # Vulcan Forged
    "GALA/USDT",  # Gala
    "ILV/USDT",   # Illuvium
    "PERP/USDT",  # Perpetual Protocol
    "CTK/USDT",   # CertiK
    "BEL/USDT",   # Bella Protocol
    "BAND/USDT",  # Band Protocol
    "NKN/USDT",   # NKN
    "CELR/USDT",  # Celer Network
    "MTL/USDT",   # Metal
    "OGN/USDT",   # Origin Protocol
    "WOO/USDT",   # WOO Network
    "DENT/USDT",  # Dent
]

# Simplified list for testing (top 20)
TOP_20_CRYPTO_SYMBOLS = TOP_100_CRYPTO_SYMBOLS[:20]

# For yfinance compatibility, convert to ticker format
def to_yfinance_ticker(symbol: str) -> str:
    """
    Convert trading pair to yfinance ticker
    BTC/USDT -> BTC-USD
    """
    base = symbol.split('/')[0]
    return f"{base}-USD"

def to_trading_pair(ticker: str) -> str:
    """
    Convert yfinance ticker to trading pair
    BTC-USD -> BTC/USDT
    """
    base = ticker.split('-')[0]
    return f"{base}/USDT"

# Symbol categories for strategic trading
SYMBOL_CATEGORIES = {
    'giants': TOP_100_CRYPTO_SYMBOLS[:10],      # Top 10 by market cap
    'majors': TOP_100_CRYPTO_SYMBOLS[10:30],    # Major alts
    'mid_cap': TOP_100_CRYPTO_SYMBOLS[30:60],   # Mid-cap coins
    'emerging': TOP_100_CRYPTO_SYMBOLS[60:80],  # Emerging projects
    'growing': TOP_100_CRYPTO_SYMBOLS[80:100],  # Growing projects
}

def get_symbols_by_category(category: str) -> list:
    """Get symbols by category"""
    return SYMBOL_CATEGORIES.get(category, [])

def get_all_symbols() -> list:
    """Get all 100 symbols"""
    return TOP_100_CRYPTO_SYMBOLS.copy()

def get_top_n_symbols(n: int = 20) -> list:
    """Get top N symbols"""
    return TOP_100_CRYPTO_SYMBOLS[:min(n, len(TOP_100_CRYPTO_SYMBOLS))]


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🪙 Top 100 Cryptocurrency Symbols")
    print("="*70 + "\n")

    print(f"Total symbols: {len(TOP_100_CRYPTO_SYMBOLS)}")
    print(f"\nTop 10 Giants:")
    for i, symbol in enumerate(TOP_100_CRYPTO_SYMBOLS[:10], 1):
        print(f"  {i:2d}. {symbol}")

    print(f"\nCategories:")
    for category, symbols in SYMBOL_CATEGORIES.items():
        print(f"  {category}: {len(symbols)} symbols")

    print("\n" + "="*70)
    print("✅ Symbol list ready!")
    print("="*70 + "\n")
