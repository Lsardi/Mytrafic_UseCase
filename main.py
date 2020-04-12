import pyspark.sql.functions as f
from pyspark.sql import Window
from pyspark.sql.types import DoubleType
from pyspark.sql import Row
from math import radians, cos, sin, asin, sqrt
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('abc').getOrCreate()
sc = spark.sparkContext

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    return km


train = (spark.read.option("inferSchema", "true")
         .option("header", "true")
         .csv("train.csv"))

distanceGPS_UDF = f.udf(haversine, DoubleType())
train = train.withColumn('time_in_min', f.col('trip_duration') / 60) \
    .withColumn('KM', distanceGPS_UDF(
    f.col('pickup_longitude'), f.col('pickup_latitude'), f.col('dropoff_longitude'), f.col('dropoff_latitude'))) \
    .withColumn('KMH_AVG', f.col('KM') * 60 / f.col('time_in_min')) \
    .withColumn('DayOfW',
                f.from_unixtime(f.unix_timestamp("pickup_datetime", "MM/dd/yyyy hh:mm:ss"), "EEEEE").alias("dow"))

df = train.groupBy('DayOfW').count().withColumnRenamed("count", "q2")

df2 = (train
       .groupBy("DayOfW")
       .agg(f.sum("KM")))
df1 = train.join(df, df.DayOfW == train.DayOfW, how='left')

row = Row("id", "start", "some_value")
tst = sc.parallelize([
    row(1, "2015-01-01 00:00:01", 20.0),
    row(1, "2015-01-01 00:00:02", 10.0),
    row(1, "2015-01-01 00:00:03", 25.0),
    row(1, "2015-01-01 00:00:04", 30.0),
    row(2, "2015-01-01 00:00:05", 5.0),
    row(2, "2015-01-01 00:00:06", 30.0),
    row(2, "2015-01-01 00:00:07", 20.0)
]).toDF().withColumn("start", f.col("start").cast("timestamp"))

w = (Window()
     .partitionBy(f.col("id"))
     .orderBy(f.col("start").cast("long"))
     .rangeBetween(-2, 0))


tst.select(f.col("*"),f.col('start').cast('long'), f.count("id").over(w).alias("mean")).show()
tst.printSchema()
