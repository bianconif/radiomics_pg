from radiomics_pg.basics import DBDriver
dbfile = '../output/test.db'
db = DBDriver(dbfile)

db.add_case('001')
db.add_case('002')
#exists = db.case_exists('001')
#exists = db.case_exists('003')
exists = db.feature_exists('Feature1')
db.add_feature('Feature1')
exists = db.feature_exists('Feature1')
exists = db.feature_exists('Feature2')
db.set_feature_value(case_id = '001', feature_name = 'Feature1', value = 0.05,
                     data_type = 'DOUBLE')
db.set_feature_value(case_id = '002', feature_name = 'Feature1', value = 0.15,
                     data_type = 'DOUBLE')
feature_value_1 = db.get_feature_value(case_id = '001', 
                                     feature_name = 'Feature1')
feature_value_2 = db.get_feature_value(case_id = '002', 
                                     feature_name = 'Feature1')
feature_value_3 = db.get_feature_value(case_id = '003', 
                                     feature_name = 'Feature1')
a = 0