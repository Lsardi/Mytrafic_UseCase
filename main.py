from pyspark import SparkContext
from pyspark.shell import spark
import pyspark.sql.functions as f
from pyspark.sql.types import *
from math import sin, cos, acos, pi


#############################################################################
def deg2rad(dd):
    """Convertit un angle "degrés décimaux" en "radians"
    """
    return dd / 180 * pi


#############################################################################
def distanceGPS(latA, longA, latB, longB):
    """Retourne la distance en KM entre les 2 points A et B connus grâce à
       leurs coordonnées GPS (en radians).
    """
    # Rayon de la terre en mètres (sphère IAG-GRS80)
    RT = 6378137

    latA = deg2rad(latA)
    longA = deg2rad(longA)

    latB = deg2rad(latB)  # Nord
    longB = deg2rad(longB)  # Est

    # angle en radians entre les 2 points
    S = acos(sin(latA) * sin(latB) + cos(latA) * cos(latB) * cos(abs(longB - longA)))
    # distance entre les 2 points, comptée sur un arc de grand cercle
    return S * RT / 1000


train = (spark.read.option("inferSchema", "true")
         .option("header", "true")
         .csv("train.csv"))

distanceGPS_UDF = f.udf(distanceGPS, DoubleType())
train = train.withColumn('time_in_min', f.col('trip_duration') / 60) \
    .withColumn('KM', distanceGPS_UDF(
    f.col('pickup_latitude'), f.col('pickup_longitude'), f.col('dropoff_latitude'), f.col('dropoff_longitude'))) \
    .withColumn('KM_AVG', f.col('KM') * 60 / f.col('time_in_min'))

df = train.groupBy('id', window('pickup_datetime', '4 hours').alias('model_window')) \
    .agg()