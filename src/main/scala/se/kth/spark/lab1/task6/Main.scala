package se.kth.spark.lab1.task6

import org.apache.spark._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import ch.systemsx.cisd.hdf5._
import java.io.File

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._

import scala.util.Try


object Main {
  def main(args: Array[String]) {
    //val conf = new SparkConf().setAppName("lab1")
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._


    val files: Vector[String] = getPaths("/home/victor/Desktop/dataset/MillionSongSubset/data/")
    val h5PathRDD = sc.parallelize(files, 5)

    val songsRDD: RDD[Row] = h5PathRDD.map(open).flatMap(_.toOption)
      .map((f: IHDF5Reader) => {

        val a = f.compounds().read("/metadata/songs", classOf[HDF5CompoundDataMap])
        val analysis  = f.compounds().read("/analysis/songs", classOf[HDF5CompoundDataMap])
        val mb  = f.compounds().read("/musicbrainz/songs", classOf[HDF5CompoundDataMap])

        val f1 = a.get("artist_familiarity")
        val f2 = a.get("song_hotttnesss")
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

        (mb.get("year"),
          analysis.get("loudness"),
          analysis.get("tempo"),
          analysis.get("danceability"),
          analysis.get("duration"),
          analysis.get("mode"),
          analysis.get("start_of_fade_out"),
          analysis.get("key"),
          analysis.get("energy"),
          analysis.get("time_signature"),
          analysis.get("end_of_fade_in"),
          analysis.get("analysis_sample_rate"),
          a.get("song_hotttnesss"))
      }).filter((r: (AnyRef, AnyRef, AnyRef, AnyRef, AnyRef, AnyRef, AnyRef, AnyRef, AnyRef, AnyRef, AnyRef, AnyRef, AnyRef)) => {
        r._1.toString.toInt > 0 && !r._13.toString.toDouble.isNaN
      }).map(r => {
        Row(
          r._1.toString.toDouble,
          r._2.toString.toDouble,
          r._3.toString.toDouble,
          r._4.toString.toDouble,
          r._5.toString.toDouble,
          r._6.toString.toInt,
          r._7.toString.toDouble,
          r._8.toString.toInt,
          r._9.toString.toDouble,
          r._10.toString.toInt,
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
            StructField("mode", IntegerType, false) ::
            StructField("start_of_fade_out", DoubleType, false) ::
            StructField("key", IntegerType, false) ::
            StructField("energy", DoubleType, false) ::
            StructField("time_signature", IntegerType, false) ::
            StructField("end_of_fade_in", DoubleType, false) ::
            StructField("analysis_sample_rate", DoubleType, false) ::
            StructField("song_hotttnesss", DoubleType, false) :: Nil)

    sqlContext.createDataFrame(songsRDD, schema).show(5)

    
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