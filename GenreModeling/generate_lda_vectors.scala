// BE SURE TO LINK SPARK-CSV:
// --packages com.databricks:spark-csv_2.11:1.2.0
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType}
import org.apache.spark.sql.Row
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.functions.max
import java.io._
import org.apache.spark.mllib.clustering.{LDA, DistributedLDAModel, LocalLDAModel}
import org.apache.spark.mllib.feature.IDF

// Dataframe schema for scrobble data
val scrobble_schema = (new StructType)
  .add("user_id", IntegerType, true)
  .add("item_id", IntegerType, true)
  .add("artist_id", IntegerType, true)
  .add("scrobble_time", StringType, true)

// Load scrobble dataframe, selecting just user and artist_id columns
val scrobbles = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("header", "false")
    .schema(scrobble_schema)
    .option("delimiter", "\t")
    .load("gs://music-foraging/scrobble_sample.txt")
    //.load("gs://music-foraging/lastfm_scrobbles.txt")
    //.load("/users/jlorince/Dropbox/sample.txt")
    .select($"user_id".cast("long").alias("user_id"), $"artist_id")
scrobbles.registerTempTable("scrobbles")

// Dataframe schema for item data
val item_schema = (new StructType)
  .add("item_id", IntegerType, true)
  .add("artist", StringType, true)
  .add("total_scrobbles", IntegerType, true)
  .add("unique_listeners", IntegerType, true)

// Load item dataframe
val items = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("header", "false")
    .schema(item_schema)
    .option("delimiter", "\t")
    .load("gs://music-foraging/artist_data")
    //.load("artist_data")
items.registerTempTable("items")

// Core join - links IDs to artist names and only selects scrobbles satisfying total listen(er) thresholds
val df = sqlContext.sql("select user_id, artist from scrobbles s join items i on s.artist_id=i.item_id where total_scrobbles>=1000 and unique_listeners>=100")

// index artist names
val indexer = new StringIndexer()
  .setInputCol("artist")
  .setOutputCol("artistIndexed")
val indexed = indexer.fit(df)
  .transform(df)

// Save artist name - index mapping to file
val vocabMap = indexed.select($"artist",$"artistIndexed").distinct().rdd.map(x=>(x(0),x(1).asInstanceOf[Double].asInstanceOf[Int])).collect()
val outfile = new File("vocab_idx")
@transient val bw = new BufferedWriter(new FileWriter(outfile))
for ((key, value) <- vocabMap) bw.write(key + "\t" + value+"\n")
bw.close()

// Group by user and artist, get counts
val indexed_agg = indexed.drop("artist")
  .withColumn("artistIndexed", $"artistIndexed".cast("integer"))
  .groupBy($"user_id", $"artistIndexed")
  .agg(count(lit(1)).alias("cnt").cast("double")).repartition(24)

// Generate paired RDD and group
val pairs = indexed_agg.map{case Row(user: Long, artist: Int, cnt: Double) => (user, (artist, cnt))}
val docs = pairs.groupByKey

// Get number of unique artists for generating sparse vectors
val n = indexed_agg.select(max($"artistIndexed")).first.getInt(0) + 1

// Represent docs as feature vectors for LDA
val docsWithFeatures = docs.mapValues(vs => Vectors.sparse(n, vs.toSeq))

// Now save to file
docsWithFeatures.saveAsTextFile("LDA_vectors")
