"""
Calculates predicted lung function (spirometry) according to the Global Lung Initiative guidelines and reference
 equations.

 Classes:
 --------
 Calculator :
    calculator class containing spline lookup table and  methods to compute FEV1, FVC, and FEV1/FVC predictions,
     z-score, and LLN.
"""

import pandas as pd
import numpy as np
from pathlib import Path

class Calculator:
    def __init__(self, file_path=None):
        """
        Creates the calculator object, used to make predictions.
        :param file_path: file path of lookup table. If None (default) then spline lookup table will be taken from package
        """
        if file_path is None:
            file_path= Path(__file__).parent / "data" / "gli_global_lookuptables_dec6.csv"

        df_lookup = pd.read_csv(file_path, index_col=['Male', 'Param', 'Age'])
        self.lookup_table = df_lookup.to_dict(orient='index')

    def predict_fev1(self, male, age, height):
        """
        Predicts a subject's healthy Forced Expiratory Volume in 1 second (FEV1).
        :param male: sex (male)
        :param age: age (years)
        :param height: standing height (cm)
        :return: predicted FEV1 (L)
        """
        #find spline values
        try:
            splines = self.lookup_table[(male, 'fev1', age)]
        except:
            raise ValueError("Participant value not present in lookup")
        m_spline = splines['M Spline']
        s_spline = splines['S Spline']
        l_spline = splines['L Spline']

        #compute reference equations
        if (male == 1):
            m = np.exp(-11.399108 + 2.462664 * np.log(height) - 0.011394 * np.log(age) + m_spline)
        elif (male == 0):
            m = np.exp(-10.901689 + 2.385928 * np.log(height) - 0.076386 * np.log(age) + m_spline)
        else:
            raise ValueError("sex (male) should be 0 or 1.")
        return m

    def predict_fvc(self, male, age, height):
        """
        Predicts a subject's healthy Forced Vital Capacity (FVC).
        :param male: sex (male)
        :param age: age (years)
        :param height: standing height (cm)
        :return: predicted FVC (L)
        """
        #find spline values
        try:
            splines = self.lookup_table[(male, 'fvc', age)]
        except:
            raise ValueError("Participant value not present in lookup")
        m_spline = splines['M Spline']
        s_spline = splines['S Spline']
        l_spline = splines['L Spline']

        #compute reference equations
        if (male == 1):
            m = np.exp(-12.629131 + 2.727421 * np.log(height) + 0.009174 * np.log(age) + m_spline)
        elif (male == 0):
            m = np.exp(-12.055901 + 2.621579 * np.log(height) - 0.035975 * np.log(age) + m_spline)
        else:
            raise ValueError("sex (male) should be 0 or 1.")
        return m

    def predict_fev1fvc(self, male, age, height):
        """
        Predicts a subject's healthy FEV1/FVC ratio.
        :param male: sex (male)
        :param age: age (years)
        :param height: standing height (cm)
        :return: predicted FEV1/FVC (unitless ratio)
        """
        #find spline values
        try:
            splines = self.lookup_table[(male, 'fev1fvc', age)]
        except:
            raise ValueError("Participant value not present in lookup")
        m_spline = splines['M Spline']
        s_spline = splines['S Spline']
        l_spline = splines['L Spline']

        #compute reference equations
        if (male == 1):
            m = np.exp(1.022608 - 0.218592 * np.log(height) - 0.027584 * np.log(age) + m_spline)
        elif (male == 0):
            m = np.exp(0.9189568 - 0.1840671 * np.log(height) - 0.0461306 * np.log(age) + m_spline)
        else:
            raise ValueError("sex (male) should be 0 or 1.")
        return m

    def zscore_fev1(self, male, age, height, measured_fev1):
        """
        for a subject with spirometry measured FEV1, computes the z-score of the measurement.
        :param male: sex (male)
        :param age: age (years)
        :param height: standing height (cm)
        :param measured_fev1: measured FEV1 (L)
        :return: fev1 z-score (1 SD)
        """
        #find spline values
        try:
            splines = self.lookup_table[(male, 'fev1', age)]
        except:
            raise ValueError("Participant value not present in lookup")
        m_spline = splines['M Spline']
        s_spline = splines['S Spline']
        l_spline = splines['L Spline']

        #compute reference equations
        if (male == 1):
            m = np.exp(-11.399108 + 2.462664 * np.log(height) - 0.011394 * np.log(age) + m_spline)
            s = np.exp(-2.256278 + 0.080729 * np.log(age) + s_spline)
            l = 1.22703
        elif (male == 0):
            m = np.exp(-10.901689 + 2.385928 * np.log(height) - 0.076386 * np.log(age) + m_spline)
            s = np.exp(-2.364047 + 0.129402 * np.log(age) + s_spline)
            l = 1.21388
        else:
            raise ValueError("sex (male) should be 0 or 1.")

        #compute z-score from reference M,S,L values:
        if (measured_fev1 is not None) and (measured_fev1>0):
            zscore = (((measured_fev1 / m) ** l) - 1) / (l * s)
        else:
            raise ValueError(f'Invalid measured_fev1:{measured_fev1}. FEV1 must be >0L')
        return zscore

    def zscore_fvc(self, male, age, height, measured_fvc):
        """
        for a subject with spirometry measured FVC, computes the z-score of the measurement.
        :param male: sex (male)
        :param age: age (years)
        :param height: standing height (cm)
        :param measured_fev1: measured FVC (L)
        :return: FVC z-score (1 SD)
        """
        # find spline values
        try:
            splines = self.lookup_table[(male, 'fvc', age)]
        except:
            raise ValueError("Participant value not present in lookup")
        m_spline = splines['M Spline']
        s_spline = splines['S Spline']
        l_spline = splines['L Spline']

        # compute reference equations
        if (male == 1):
            m = np.exp(-12.629131 + 2.727421 * np.log(height) + 0.009174 * np.log(age) + m_spline)
            s = np.exp(-2.195595 + 0.068466 * np.log(age) + s_spline)
            l = 0.9346
        elif (male == 0):
            m = np.exp(-12.055901 + 2.621579 * np.log(height) - 0.035975 * np.log(age) + m_spline)
            s = np.exp(-2.310148 + 0.120428 * np.log(age) + s_spline)
            l = 0.89900
        else:
            raise ValueError("sex (male) should be 0 or 1.")

        # compute z-score from reference M,S,L values:
        if (measured_fvc is not None) and (measured_fvc > 0):
            zscore = (((measured_fvc / m) ** l) - 1) / (l * s)
        else:
            raise ValueError(f'Invalid measured_fvc:{measured_fvc}. FVC must be >0L')
        return zscore
    
    def zscore_fev1fvc(self, male, age, height, measured_fev1fvc):
        """
        for a subject with spirometry measured FEV1/FVC, computes the z-score of the measurement.
        :param male: sex (male)
        :param age: age (years)
        :param height: standing height (cm)
        :param measured_fev1: measured FEV1/FVC
        :return: FEV1/FVC z-score (1 SD)
        """
        # find spline values
        try:
            splines = self.lookup_table[(male, 'fev1fvc', age)]
        except:
            raise ValueError("Participant value not present in lookup")
        m_spline = splines['M Spline']
        s_spline = splines['S Spline']
        l_spline = splines['L Spline']

        # compute reference equations
        if (male == 1):
            m = np.exp(1.022608 - 0.218592 * np.log(height) - 0.027584 * np.log(age) + m_spline)
            s = np.exp(-2.882024 + 0.068889 * np.log(age) + s_spline)
            l = 3.8243 - 0.3328 * np.log(age)
        elif (male == 0):
            m = np.exp(0.9189568 - 0.1840671 * np.log(height) - 0.0461306 * np.log(age) + m_spline)
            s = np.exp(-3.171582 + 0.144358 * np.log(age) + s_spline)
            l = 6.6490 - 0.9920 * np.log(age)
        else:
            raise ValueError("sex (male) should be 0 or 1.")

        # compute z-score from reference M,S,L values:
        if (measured_fev1fvc is not None) and (measured_fev1fvc > 0):
            zscore = (((measured_fev1fvc / m) ** l) - 1) / (l * s)
        else:
            raise ValueError(f'Invalid measured_fev1fvc:{measured_fev1fvc}. FEV1/FVC must be >0L')
        return zscore

