package se.kth.spark.lab1.task1
import org.apache.spark.sql.functions.{min, max, mean}

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    //val rawDF = sqlContext.read.text(filePath)

    val rdd = sc.textFile(filePath)

    //Step1: print the first 5 rows, what is the delimiter, number of features and the data types
    rdd.top(5).foreach(println)

    //Step2: split each row into an array of features
    val recordsRdd = rdd.map(_.split(","))

    //Step3: map each row into a Song object by using the year label and the first three features  
    val songsRdd = recordsRdd.map((features: Array[String]) => {
      (features(0).replace(".0", "").toInt, features(1).toDouble, features(2).toDouble)
    })

    //Step4: convert your rdd into a dataframe
    val songsDf = songsRdd.toDF("year", "feature1", "feature2")
    songsDf.show(5)

    //Question 1
    println("Total count: " + songsDf.count())

    //Question 2
    val nrBetween98And00 = songsRdd.filter{case(y, _, _) => { y >= 1998 && y < 2000}}.count()
    println("Nr of songs between 1998 and 2000: " + nrBetween98And00)

    //Question 3
    songsDf.agg(min("year").as("min"), max("year").as("max"), mean("year").as("mean")).show

    //Question 4
    val songCountRdd = songsRdd.filter{case(y, _, _) => { y >= 2000 && y < 2010}}
        .groupBy{case(y, _, _) => y}
        .map{case (y:Int, iter:Iterable[(Int, _, _)]) => {

          (y, iter.size)
        }}.sortBy(f => f._1).foreach(f => {
          println("Year: " + f._1 + " => " + f._2)
        })
  }
}