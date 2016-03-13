//spark-shell --executor-cores 2 --conf spark.serializer=org.apache.spark.serializer.KryoSerializer
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


// If we're loading from text
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

def parse(rdd: RDD[String]): RDD[(Long, Vector)] = {
  val pattern: scala.util.matching.Regex = "\\(([0-9]+),(.*)\\)".r
  rdd .map{
    case pattern(k, v) => (k.toLong, Vectors.parse(v))
  }
}

val text = sc.textFile("gs://music-foraging/LDA_vectors")
//val text = sc.textFile("gs://music-foraging/LDA_vectors_tfidf")
//val docsWithFeatures = parse(text).repartition(96).cache()
val docsWithFeatures = parse(text).cache()


// Main LDA loop
for (n_topics <- Range(195,206,5) ){

    val s = n_topics.toString
    val lda = new LDA().setK(n_topics).setMaxIterations(50) // .setAlpha(), .setBeta()

    val ldaModel = lda.run(docsWithFeatures)

    val distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]
    //val localLDAModel = distLDAModel.toLocal


      val outfile = new File(s"model_summary_$s")
      @transient val bw = new BufferedWriter(new FileWriter(outfile))
      bw.write("topicConcentration:"+"\t"+distLDAModel.topicConcentration+"\n")
      bw.write("docConcentration:"+"\t"+distLDAModel.docConcentration(0)+"\n")
      bw.write("LL:"+"\t"+distLDAModel.logLikelihood+"\n")
      bw.close()

      distLDAModel.topicDistributions.saveAsTextFile(s"user_topic_$s")

      val topic_mat = distLDAModel.topicsMatrix

      val localMatrix: List[Array[Double]] = topic_mat.transpose.toArray.grouped(topic_mat.numCols).toList

      val lines: List[String] = localMatrix.map(line => line.mkString(" "))

      val outfile2 = new File(s"artist_topic_$s")
      @transient val bw2 = new BufferedWriter(new FileWriter(outfile2))
      for (line <- lines) bw2.write(line+"\n")
      bw2.close()
  }
