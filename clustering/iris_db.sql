use iris_db;

SELECT petal_length, petal_width, sepal_length, sepal_width, species_id, species_name
FROM measurements m
JOIN species s USING(species_id);

