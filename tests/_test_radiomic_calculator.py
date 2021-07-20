import json

from radiomics_pg.basics import RadiomicCalculator

config_file = '../config-files/compute-features-CT.json'
with open(config_file) as fp:
    config = json.load(fp)

#Create the calculator
calculator = RadiomicCalculator(config_file)

#Compute the features
case_list = calculator.get_cases()
features_to_compute = config['FeaturesToCompute']
for case in case_list:
    for feature in features_to_compute:
        calculator.get_feature_value(case, feature[0], feature[1])
a = 0