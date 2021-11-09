import os
import pandas as pd
import numpy as np
from functools import reduce

os.chdir("/Users/sophiemerl/Desktop/Data Analytics in Business/PythonProject")
path = os.getcwd()

######### CASES
file = "Cases_Specimendate.csv"
cases = pd.read_csv(file)
cases = cases.drop(['areaName',
                    'areaType',
                    'areaCode'], axis=1)

cases = cases.replace(np.nan, -999).rename(columns={"date": "Date"})

cases['Date'] = pd.to_datetime(cases['Date'], format='%Y-%m-%d')

######### VACCINATION
file = "Vaccination.csv"
vaccination = pd.read_csv(file)
vaccination = vaccination.drop(['areaName',
                                'areaType',
                                'areaCode'], axis=1)

vaccination = vaccination.replace(np.nan, -999).rename(columns={"date": "Date"})
vaccination['Date'] = pd.to_datetime(vaccination['Date'], format='%Y-%m-%d')

######### TESTING & POSITIVITY

file = "Testing & Positivity.csv"
testing_positivity = pd.read_csv(file)
testing_positivity = testing_positivity.drop(['areaName',
                                              'areaType',
                                              'areaCode'], axis=1)

testing_positivity = testing_positivity.replace(np.nan, -999).rename(columns={"date": "Date"})
testing_positivity['Date'] = pd.to_datetime(testing_positivity['Date'], format='%Y-%m-%d')

######### HEALTHCARE
file1 = "healthcare_patientsventilation.csv"
file2 = "healthcare_patientshospital.csv"
file3 = "healthcare_patientsadmitted.csv"

ventilation = pd.read_csv(file1)
patients_hospital = pd.read_csv(file2)
patients_admitted = pd.read_csv(file3)
ventilation = ventilation.drop(['areaName',
                                'areaType',
                                'areaCode'],
                               axis=1).replace(np.nan, -999).rename(columns={"date": "Date"})
patients_hospital = patients_hospital.drop(['areaName',
                                            'areaType',
                                            'areaCode'],
                                           axis=1).replace(np.nan, -999).rename(columns={"date": "Date"})
patients_admitted = patients_admitted.drop(['areaName',
                                   'areaType',
                                   'areaCode'],
                                           axis=1).replace(np.nan, -999).rename(columns={"date": "Date"})

healthcare = [ventilation,
              patients_hospital,
              patients_admitted]

healthcare = reduce(lambda  left,right: pd.merge(left,right,on=['Date'],
                                            how='outer'), healthcare)
healthcare['Date'] = pd.to_datetime(healthcare['Date'], format='%Y-%m-%d')

######### DEATHS
file = "deaths.csv"
deaths = pd.read_csv(file)
deaths = deaths.drop(['areaName',
                      'areaType',
                      'areaCode'], axis=1)

deaths = deaths.replace(np.nan, -999).rename(columns={"date": "Date"})
deaths['Date'] = pd.to_datetime(deaths['Date'], format='%Y-%m-%d')

######### MERGE
data_sets = [cases,
             vaccination,
             testing_positivity,
             healthcare,
             deaths]

data_sets_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Date'],
                                            how='outer'), data_sets)

data_sets_merged.to_csv(r'London_Covid_301021.csv')
