package se.kth.spark.lab1.task6

import org.apache.spark._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import ch.systemsx.cisd.hdf5._
import java.io.File

import org.apache.spark.ml.feature.{Normalizer, StandardScaler}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import se.kth.spark.lab1.Array2Vector

import scala.util.Try


object Main {
  def main(args: Array[String]) {
    //val conf = new SparkConf().setAppName("lab1")
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._


    val files: Vector[String] = getPaths("D:\\tmp\\millionsongsubset_full.tar\\millionsongsubset_full\\MillionSongSubset\\data\\")
    val h5PathRDD = sc.parallelize(files, 5)

    val songsRDD: RDD[Row] = h5PathRDD.map(open).flatMap(_.toOption)
      .map((f: IHDF5Reader) => {

        val a = f.compounds().read("/metadata/songs", classOf[HDF5CompoundDataMap])
        val analysis  = f.compounds().read("/analysis/songs", classOf[HDF5CompoundDataMap])
        val mb  = f.compounds().read("/musicbrainz/songs", classOf[HDF5CompoundDataMap])

        /*
        (mb.get("year"),
          analysis.get("mode_confidence"),
          analysis.get("loudness"),
          analysis.get("tempo"),
          analysis.get("idx_bars_start"),
          analysis.get("danceability"),
          analysis.get("idx_beats_confidence"),
          analysis.get("idx_tatums_confidence"),
          analysis.get("key_confidence"),
          analysis.get("idx_segments_pitches"),
          analysis.get("duration"),
          analysis.get("mode"),
          analysis.get("idx_beats_start"),
          analysis.get("start_of_fade_out"),
          analysis.get("idx_sections_start"),
          analysis.get("idx_segments_loudness_start"),
          analysis.get("time_signature_confidence"),
          analysis.get("key"),
          analysis.get("energy"),
          analysis.get("time_signature"),
          analysis.get("idx_segments_timbre"),
          analysis.get("idx_segments_start"),
          analysis.get("end_of_fade_in"),
          analysis.get("idx_segments_loudness_max"),
          analysis.get("idx_segments_loudness_max_time"),
          analysis.get("analysis_sample_rate"),
          analysis.get("idx_segments_confidence"),
          analysis.get("track_id"),
          analysis.get("idx_bars_confidence"),
          analysis.get("idx_tatums_start"),
          analysis.get("idx_sections_confidence"))
          */
        val f1 = mb.get("year")
        val f2 = analysis.get("loudness")
        val f3 = analysis.get("tempo")
        val f4 = analysis.get("danceability")
        val f5 = analysis.get("duration")
        val f6 = analysis.get("mode")
        val f7 = analysis.get("start_of_fade_out")
        val f8 = analysis.get("key")
        val f9 = analysis.get("energy")
        val f10 = analysis.get("time_signature")
        val f11= analysis.get("end_of_fade_in")
        val f12 =  analysis.get("analysis_sample_rate")
        val f13= a.get("song_hotttnesss")
        f.close()
        (f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13)
      })
      .filter((r: (AnyRef, AnyRef, AnyRef, AnyRef, AnyRef, AnyRef, AnyRef, AnyRef, AnyRef, AnyRef, AnyRef, AnyRef, AnyRef)) => {
        r._1.toString.toInt > 0 && !r._13.toString.toDouble.isNaN
      }).map(r => {
        Row(
          r._1.toString.toDouble,
          r._2.toString.toDouble,
          r._3.toString.toDouble,
          r._4.toString.toDouble,
          r._5.toString.toDouble,
          r._6.toString.toDouble,
          r._7.toString.toDouble,
          r._8.toString.toDouble,
          r._9.toString.toDouble,
          r._10.toString.toDouble,
          r._11.toString.toDouble,
          r._12.toString.toDouble,
          r._13.toString.toDouble)
      })

    val schema = StructType(
          StructField("year", DoubleType, false) ::
            StructField("loudness", DoubleType, false) ::
            StructField("tempo", DoubleType, false) ::
            StructField("danceability", DoubleType, false) ::
            StructField("duration", DoubleType, false) ::
            StructField("mode", DoubleType, false) ::
            StructField("start_of_fade_out", DoubleType, false) ::
            StructField("key", DoubleType, false) ::
            StructField("energy", DoubleType, false) ::
            StructField("time_signature", DoubleType, false) ::
            StructField("end_of_fade_in", DoubleType, false) ::
            StructField("analysis_sample_rate", DoubleType, false) ::
            StructField("song_hotttnesss", DoubleType, false) :: Nil)

    val songsDF = sqlContext.createDataFrame(songsRDD, schema)

    //Normalize
    val featuresRDD = songsDF.rdd.map{row =>
//      (row.get(0), row.get(1), row.get(2), row.get(3), row.get(4), row.get(5),
//      row.get(6), row.get(7), row.get(8), row.get(9), row.get(10), row.get(11), row.get(12))
      (row.toSeq.head, row.toSeq.tail)
    }

    val schema2 = StructType(
      StructField("label", DoubleType, false) ::
        StructField("fields", ArrayType(StringType), false) :: Nil)


    val featuresRDDforNormalization =
      featuresRDD.map(r => {
      Row(
        r._1.toString.toDouble,
        r._2.map(item => item.toString))
    })

    val featuresDFforNormalization = sqlContext.createDataFrame(featuresRDDforNormalization, schema2)

    featuresDFforNormalization.take(5).foreach(println)

    val arr2Vect = new Array2Vector()
    val vectorTr = arr2Vect
      .setInputCol("fields")
      .setOutputCol("features")

    val vectorDF = vectorTr.transform(featuresDFforNormalization)
    vectorDF.cache()

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)

    // Compute summary statistics by fitting the StandardScaler
    val scalerModel = scaler.fit(vectorDF)

    // Normalize each feature to have unit standard deviation.
    val scaledData = scalerModel.transform(vectorDF)
    scaledData.take(5).foreach(println)

    /*
    val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(2.0)

    val l1NormData = normalizer.transform(vectorDF)
    l1NormData.take(5).foreach(println)
    */

    val songDFforPipeline = scaledData.select("label", "scaledFeatures").rdd.map{row =>
      row.get(0) + "," + row.get(1).asInstanceOf[DenseVector].toArray.foldRight(z = "")((z, a1) => if (a1.nonEmpty) z+","+a1 else z+"")
    }

    songDFforPipeline.take(5).foreach(println)

    val splits = songDFforPipeline.toDF("row").randomSplit(Array(0.7, 0.3))

    val obsDF = splits(0)
    val testDF = splits(1)

    val pipeline = PipelineBuilder.build(obsDF)

    val pipelineModel: PipelineModel = pipeline.fit(obsDF)
    val lReg = pipelineModel.stages(7).asInstanceOf[MyLinearModelImpl]
    lReg.trainingError.foreach(e => {
      println("RMSE => " + e)
    })

    lReg.transform(pipelineModel.transform(testDF).select("row", "features")).show(5)
  }



  def open(filename: String): Try[IHDF5Reader] = Try{HDF5FactoryProvider.get().openForReading(new File(filename))}

  def getPaths(path: String): Vector[String] = {
    val dir = new java.io.File(path)
    getFiles(dir).map(_.getAbsolutePath)
  }

  // Retrieve collection of all files within this directory
  def getFiles(dir: File): Vector[File] = {
    val dirs = collection.mutable.Stack[File]()
    val these = collection.mutable.ArrayBuffer[File]()
    dirs.push(dir)

    while (dirs.nonEmpty) {
      val dir = dirs.pop()
      val children = dir.listFiles
      val files = children.filterNot(_.isDirectory)
      val subDirectories = children.filter(_.isDirectory)
      these ++= files
      dirs.pushAll(subDirectories)
    }
    these.result().toVector
  }
}