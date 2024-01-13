import pytest
from spiropredict import calculators as calc
# Fixture for Calculator instance
@pytest.fixture
def calculator():
    return calc.Calculator()

def test_predict_fev1(calculator):
    test_male = 0
    test_age = 95
    test_height = 190
    expected_fev1 = 2.478
    result = calculator.predict_fev1(test_male, test_age, test_height)
    assert round(result, 2) == round(expected_fev1, 2)

def test_predict_fvc(calculator):
    test_male = 1
    test_age = 75
    test_height = 170
    expected_fvc =  3.471
    result = calculator.predict_fvc(test_male, test_age, test_height)
    assert round(result, 2) == round(expected_fvc, 2)

def test_predict_fev1fvc(calculator):
    test_male = 0
    test_age = 50
    test_height = 150
    expected_fev1fvc =  0.824
    result = calculator.predict_fev1fvc(test_male, test_age, test_height)
    assert round(result, 2) == round(expected_fev1fvc, 2)

def test_zscore_fev1(calculator):
    test_male = 0
    test_age = 95
    test_height = 190
    test_fev1 = 2.478
    expected_zscore = 0
    result = calculator.zscore_fev1(test_male, test_age, test_height, test_fev1)
    assert round(result, 2) == round(expected_zscore, 2)

def test_zscore_fvc(calculator):
    test_male = 1
    test_age = 75
    test_height = 170
    test_fvc = 3.471
    expected_zscore=0
    result = calculator.zscore_fvc(test_male, test_age, test_height, test_fvc)
    assert round(result, 2) == round(expected_zscore, 2)

def test_zscore_fev1fvc(calculator):
    test_male = 0
    test_age = 50
    test_height = 150
    test_fev1fvc = 0.824
    expected_zscore=0
    result = calculator.zscore_fev1fvc(test_male, test_age, test_height, test_fev1fvc)
    assert round(result, 2) == round(expected_zscore, 2)
def test_invalid_input(calculator):
    with pytest.raises(ValueError):
        calculator.predict_fev1(2, 30, 170)  # An invalid gender value

