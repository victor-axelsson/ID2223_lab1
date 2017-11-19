package se.kth.spark.lab1.task6

import org.apache.spark._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.{DataFrame, SQLContext}
import ch.systemsx.cisd.hdf5.{HDF5Factory, HDF5FactoryProvider, IHDF5Reader, IHDF5SimpleReader}
import java.io.File

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
    //val files: Vector[String] = getPaths("hdfs:///Projects/labs/million_song/huge_dataset/")
    val h5PathRDD = sc.parallelize(files, 5)

    val songsRDD = h5PathRDD.map(open).flatMap(_.toOption)
      .map((f: IHDF5Reader) => {
        println(f)
      }).collect()

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