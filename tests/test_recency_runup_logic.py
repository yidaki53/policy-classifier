from scripts.analyze_recency_weighted_trends import _election_years_between, _is_election_runup_year


def test_election_years_between_four_year_cadence():
    years = _election_years_between(min_year=2014, max_year=2026, cadence_years=4, anchor_year=2010)
    assert years == [2014, 2018, 2022, 2026]


def test_runup_year_marks_prior_and_election_year():
    election_years = [2018, 2022]
    assert _is_election_runup_year(2017, election_years, runup_years=1)
    assert _is_election_runup_year(2018, election_years, runup_years=1)
    assert not _is_election_runup_year(2016, election_years, runup_years=1)


def test_runup_year_zero_means_election_year_only():
    election_years = [2018, 2022]
    assert _is_election_runup_year(2018, election_years, runup_years=0)
    assert not _is_election_runup_year(2017, election_years, runup_years=0)
