package se.kth.spark.lab1.task6

import org.apache.spark._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.spark.input.PortableDataStream

import scala.util.Try
import java.nio.charset._

import org.apache.spark.rdd.RDD


object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    //val conf = new SparkConf().setAppName("lab1")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    //val filePath = "src/main/resources/millionsong.txt"
    //val filePath = "hdfs:///Projects/labs/million_song/huge_dataset/*.tar.gz"
    val filePath = "/home/victor/Desktop/dataset/fakeData/*.tar.gz"

    val reader = HDF5Factory.openForReading("farray.h5")
    val mydata = reader.readFloatArray("mydata")
    /*
    val splits = sc.binaryFiles(filePath)
      .flatMapValues(x => extractFiles(x).toOption)
      .mapValues(_.map(decode()))
      .flatMap((file: (String, Array[String])) => {
        println()
      }).map((r: String) => {
        r.split("\n")
      }).flatMap((r: Array[String]) => {
        r
      }).toDF("row").randomSplit(Array(0.7, 0.3))
  */

    val splits = sc.binaryFiles(filePath)
      .flatMapValues(x => extractFiles(x).toOption)
      .mapValues(_.map(decode())).take(4).foreach(r => {
        println(r)
      })

    //val splits = sqlContext.read.text(filePath).toDF("row").randomSplit(Array(0.7, 0.3))
  /*
    val obsDF = splits(0)
    val testDF = splits(1)

    val pipeline = PipelineBuilder.build(obsDF)

    val pipelineModel: PipelineModel = pipeline.fit(obsDF)
    val lReg = pipelineModel.stages(7).asInstanceOf[MyLinearModelImpl]
    lReg.trainingError.foreach(e => {
      println("RMSE => " + e)
    })

    lReg.transform(pipelineModel.transform(testDF).select("row", "features")).show(5)
    */
  }


  def extractFiles(ps: PortableDataStream, n: Int = 1024) = Try {
    val tar = new TarArchiveInputStream(new GzipCompressorInputStream(ps.open))
    Stream.continually(Option(tar.getNextTarEntry))
      // Read until next exntry is null
      .takeWhile(_.isDefined)
      // flatten
      .flatMap(x => x)
      // Drop directories
      .filter(!_.isDirectory)
      .map(e => {
        Stream.continually {
          // Read n bytes
          val buffer = Array.fill[Byte](n)(-1)
          val i = tar.read(buffer, 0, n)
          (i, buffer.take(i))}
          // Take as long as we've read something
          .takeWhile(_._1 > 0)
          .map(_._2)
          .flatten
          .toArray})
      .toArray
  }

  def decode(charset: Charset = StandardCharsets.UTF_8)(bytes: Array[Byte]) =
    new String(bytes, StandardCharsets.UTF_8)

}