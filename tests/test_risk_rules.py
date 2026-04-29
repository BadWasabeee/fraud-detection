from risk_rules import label_risk, score_transaction


# Clean low-risk baseline. Each per-rule test changes one field so the test
# isolates that signal. If a rule ever gets re-inverted, exactly that test fails.
BASE_TX = {
    "device_risk_score": 10,
    "is_international": 0,
    "amount_usd": 25.0,
    "velocity_24h": 1,
    "failed_logins_24h": 0,
    "prior_chargebacks": 0,
}


def _tx(**overrides):
    return {**BASE_TX, **overrides}


# --- label_risk thresholds --------------------------------------------------

def test_label_risk_thresholds():
    assert label_risk(10) == "low"
    assert label_risk(35) == "medium"
    assert label_risk(75) == "high"


def test_label_risk_low_band_boundaries():
    assert label_risk(0) == "low"
    assert label_risk(29) == "low"


def test_label_risk_medium_band_boundaries():
    assert label_risk(30) == "medium"
    assert label_risk(59) == "medium"


def test_label_risk_high_band_boundaries():
    assert label_risk(60) == "high"
    assert label_risk(100) == "high"


# --- baseline ---------------------------------------------------------------

def test_clean_baseline_scores_zero_and_labels_low():
    score = score_transaction(_tx())
    assert score == 0
    assert label_risk(score) == "low"


# --- regression guards: each rule must add risk in the correct direction ----
# These would have caught the historical inversions in device_risk_score,
# is_international, velocity_24h, and prior_chargebacks.

def test_high_device_risk_increases_score():
    assert score_transaction(_tx(device_risk_score=85)) > score_transaction(_tx())


def test_medium_device_risk_increases_score():
    assert score_transaction(_tx(device_risk_score=50)) > score_transaction(_tx())


def test_international_transaction_increases_score():
    assert score_transaction(_tx(is_international=1)) > score_transaction(_tx())


def test_high_velocity_increases_score():
    assert score_transaction(_tx(velocity_24h=8)) > score_transaction(_tx())


def test_medium_velocity_increases_score():
    assert score_transaction(_tx(velocity_24h=4)) > score_transaction(_tx())


def test_repeat_prior_chargebacks_increase_score():
    assert score_transaction(_tx(prior_chargebacks=3)) > score_transaction(_tx())


def test_single_prior_chargeback_increases_score():
    assert score_transaction(_tx(prior_chargebacks=1)) > score_transaction(_tx())


def test_high_failed_logins_increase_score():
    assert score_transaction(_tx(failed_logins_24h=6)) > score_transaction(_tx())


def test_medium_failed_logins_increase_score():
    assert score_transaction(_tx(failed_logins_24h=3)) > score_transaction(_tx())


def test_large_amount_increases_score():
    assert score_transaction(_tx(amount_usd=1500)) > score_transaction(_tx())


def test_medium_amount_increases_score():
    assert score_transaction(_tx(amount_usd=750)) > score_transaction(_tx())


def test_repeat_chargebacks_score_higher_than_single():
    assert score_transaction(_tx(prior_chargebacks=3)) > score_transaction(_tx(prior_chargebacks=1))


def test_high_device_score_higher_than_medium_device():
    assert score_transaction(_tx(device_risk_score=85)) > score_transaction(_tx(device_risk_score=50))


# --- realistic combination --------------------------------------------------

def test_textbook_fraud_combination_is_high():
    # Mirrors real chargeback row 50003 in data/: high device risk, international,
    # large amount, high velocity, failed logins, one prior chargeback.
    # Pre-fix this scored 0/`low` because three signals were inverted.
    tx = _tx(
        device_risk_score=81,
        is_international=1,
        amount_usd=1250,
        velocity_24h=6,
        failed_logins_24h=5,
        prior_chargebacks=1,
    )
    score = score_transaction(tx)
    assert score >= 60
    assert label_risk(score) == "high"


# --- bounds -----------------------------------------------------------------

def test_score_is_clamped_to_100():
    tx = _tx(
        device_risk_score=99,
        is_international=1,
        amount_usd=10_000,
        velocity_24h=20,
        failed_logins_24h=10,
        prior_chargebacks=10,
    )
    assert score_transaction(tx) == 100


def test_score_never_negative():
    assert score_transaction(_tx()) >= 0


# --- legacy test kept for stability ----------------------------------------

def test_large_amount_adds_risk():
    tx = {
        "device_risk_score": 10,
        "is_international": 0,
        "amount_usd": 1200,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    assert score_transaction(tx) >= 25
