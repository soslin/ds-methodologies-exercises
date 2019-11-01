USE zillow;

#Count how many transactions in 2017
SELECT COUNT(*) FROM predictions_2017;
#77,614

#Count the number of unique properties sold in 2017
SELECT count(DISTINCT parcelid) FROM predictions_2017;
#77,414

#Join all tables, count is 77,614 - 200 hundred dups will need to be removed in Python
SELECT * FROM predictions_2017 AS pr
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
WHERE latitude NOT IS NULL AND longitude NOT IS NULL;





    
    

