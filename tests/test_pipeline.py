import pandas as pd

from analyze_fraud import score_transactions, summarize_results
from features import build_model_frame


# Schema helpers so test rows match the real CSVs and stay readable.

ACCOUNT_COLS = [
    "account_id", "customer_name", "country", "signup_date", "kyc_level",
    "account_age_days", "prior_chargebacks", "is_vip",
]
TRANSACTION_COLS = [
    "transaction_id", "account_id", "timestamp", "amount_usd", "merchant_category",
    "channel", "device_risk_score", "ip_country", "is_international",
    "velocity_24h", "failed_logins_24h", "chargeback_within_60d",
]


def _accounts(rows):
    return pd.DataFrame(rows, columns=ACCOUNT_COLS)


def _transactions(rows):
    return pd.DataFrame(rows, columns=TRANSACTION_COLS)


# --- build_model_frame ------------------------------------------------------

def test_build_model_frame_joins_account_prior_chargebacks():
    transactions = _transactions([
        (50001, 1001, "2026-02-01 00:00:00", 100.0, "grocery", "web", 10, "US", 0, 1, 0, 0),
    ])
    accounts = _accounts([
        (1001, "Alice", "US", "2024-01-01", "full", 365, 3, "N"),
    ])

    df = build_model_frame(transactions, accounts)

    assert df.loc[0, "prior_chargebacks"] == 3


def test_build_model_frame_fills_missing_account_with_zero_prior_chargebacks():
    # Defends against silent NaN -> rule no-op when an account_id is unknown.
    transactions = _transactions([
        (50001, 9999, "2026-02-01 00:00:00", 100.0, "grocery", "web", 10, "US", 0, 1, 0, 0),
    ])
    accounts = _accounts([
        (1001, "Alice", "US", "2024-01-01", "full", 365, 0, "N"),
    ])

    df = build_model_frame(transactions, accounts)

    assert df.loc[0, "prior_chargebacks"] == 0


def test_build_model_frame_preserves_transaction_count():
    transactions = _transactions([
        (50001, 1001, "2026-02-01 00:00:00", 50.0, "grocery", "web", 10, "US", 0, 1, 0, 0),
        (50002, 1001, "2026-02-01 00:00:00", 60.0, "grocery", "web", 10, "US", 0, 1, 0, 0),
    ])
    accounts = _accounts([
        (1001, "Alice", "US", "2024-01-01", "full", 365, 0, "N"),
    ])

    df = build_model_frame(transactions, accounts)

    assert len(df) == 2


# --- score_transactions (end-to-end) ---------------------------------------

def test_score_transactions_labels_clean_low_and_textbook_high():
    transactions = _transactions([
        # Clean baseline -> low.
        (50001, 1001, "2026-02-01 00:00:00", 50.0, "grocery", "web", 5, "US", 0, 1, 0, 0),
        # Textbook fraud combo -> high.
        (50002, 1002, "2026-02-01 00:00:00", 1500.0, "gift_cards", "web", 85, "PH", 1, 8, 6, 0),
    ])
    accounts = _accounts([
        (1001, "Alice", "US", "2024-01-01", "full", 365, 0, "N"),
        (1002, "Bob", "US", "2024-01-01", "full", 365, 2, "N"),
    ])

    scored = score_transactions(transactions, accounts)

    assert set(scored["risk_label"]).issubset({"low", "medium", "high"})
    by_id = scored.set_index("transaction_id")
    assert by_id.loc[50001, "risk_label"] == "low"
    assert by_id.loc[50002, "risk_label"] == "high"


def test_score_transactions_score_within_bounds():
    transactions = _transactions([
        (50001, 1001, "2026-02-01 00:00:00", 50.0, "grocery", "web", 5, "US", 0, 1, 0, 0),
        (50002, 1002, "2026-02-01 00:00:00", 9999.0, "gift_cards", "web", 99, "PH", 1, 20, 10, 0),
    ])
    accounts = _accounts([
        (1001, "Alice", "US", "2024-01-01", "full", 365, 0, "N"),
        (1002, "Bob", "US", "2024-01-01", "full", 365, 5, "N"),
    ])

    scored = score_transactions(transactions, accounts)

    assert scored["risk_score"].min() >= 0
    assert scored["risk_score"].max() <= 100


# --- summarize_results ------------------------------------------------------

def test_summarize_results_computes_chargeback_rate_per_label():
    scored = pd.DataFrame([
        {"transaction_id": 1, "amount_usd": 100.0, "risk_label": "high"},
        {"transaction_id": 2, "amount_usd": 200.0, "risk_label": "high"},
        {"transaction_id": 3, "amount_usd": 50.0,  "risk_label": "low"},
    ])
    chargebacks = pd.DataFrame([
        {"transaction_id": 1, "chargeback_date": "2026-03-15",
         "chargeback_reason": "card_not_present", "loss_amount_usd": 100.0},
    ])

    summary = summarize_results(scored, chargebacks).set_index("risk_label")

    assert summary.loc["high", "transactions"] == 2
    assert summary.loc["high", "chargebacks"] == 1
    assert summary.loc["high", "chargeback_rate"] == 0.5
    assert summary.loc["low", "chargebacks"] == 0
    assert summary.loc["low", "chargeback_rate"] == 0.0


def test_summarize_results_handles_no_chargebacks():
    scored = pd.DataFrame([
        {"transaction_id": 1, "amount_usd": 100.0, "risk_label": "low"},
        {"transaction_id": 2, "amount_usd": 200.0, "risk_label": "low"},
    ])
    chargebacks = pd.DataFrame(columns=["transaction_id", "chargeback_date",
                                        "chargeback_reason", "loss_amount_usd"])

    summary = summarize_results(scored, chargebacks).set_index("risk_label")

    assert summary.loc["low", "chargebacks"] == 0
    assert summary.loc["low", "chargeback_rate"] == 0.0
