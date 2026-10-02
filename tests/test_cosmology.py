import pytest

from nba.cosmology import Cosmology


def test_van_der_marel_2012_example():
    """Appendix example of van der Marel et al. 2012: cvir = 10."""
    cosmo = Cosmology()
    cvir = 10
    c200 = cosmo.cvirc200(cvir=cvir)
    assert c200 == pytest.approx(7.4, abs=0.05)
    assert cosmo.m200mvir(c200=c200, cvir=cvir) == pytest.approx(0.84, abs=0.01)
    assert cosmo.ars(c200) == pytest.approx(2.01, abs=0.01)
    assert cosmo.mhmvir(cosmo.ars(c200), cvir) == pytest.approx(1.36, abs=0.02)
    assert cosmo.ars(cvir) == pytest.approx(2.09, abs=0.01)
