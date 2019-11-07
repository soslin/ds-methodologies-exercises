import numpy as np
import pandas as pd
from env import host, user, password

# Data import
url = f'mysql+pymysql://{user}:{password}@{host}/zillow'


def acquire_zillow():
    zillow_data = pd.read_sql('''SELECT * FROM predictions_2017 AS pr
INNER JOIN properties_2017 as p
	ON pr.id = p.id
LEFT JOIN airconditioningtype AS a
	ON p.airconditioningtypeid = a.airconditioningtypeid
LEFT JOIN architecturalstyletype as ar
	ON ar.architecturalstyletypeid = p.architecturalstyletypeid
LEFT JOIN buildingclasstype as b
	ON b.buildingclasstypeid = p.buildingclasstypeid
LEFT JOIN heatingorsystemtype as h
	ON h.heatingorsystemtypeid = p.heatingorsystemtypeid
LEFT JOIN propertylandusetype AS plu
	ON plu.propertylandusetypeid = p.propertylandusetypeid
LEFT JOIN storytype AS s
	ON s.storytypeid = p.storytypeid
LEFT JOIN typeconstructiontype AS t
	ON t.typeconstructiontypeid = p.typeconstructiontypeid
LEFT JOIN  unique_properties as u
	ON u.parcelid = p.parcelid
WHERE plu.propertylandusetypeid IN (261,262,273,275,279)
    AND (bathroomcnt > 0 AND bedroomcnt > 0);''', url)
    return zillow_data
acquire_zillow()

df = acquire_zillow()


url2 = f'mysql+pymysql://{user}:{password}@{host}/iris_db'
def acquire_iris_data():
    iris_data = pd.read_sql('''
    SELECT petal_length, petal_width, sepal_length, sepal_width, species_id, species_name
	FROM measurements m
	JOIN species s USING(species_id); ''', url2)
    return iris_data

df2 = acquire_iris_data()


url3 = f'mysql+pymysql://{user}:{password}@{host}/mall_customers'
def acquire_mallcustomer_data():
    mallcustomer_data = pd.read_sql('''SELECT * FROM customers;''', url3)
    return mallcustomer_data

df3 = acquire_mallcustomer_data()
