package se.kth.spark.lab1.task6

import org.apache.spark._
import org.apache.spark.ml.{PipelineModel}
import org.apache.spark.sql.{DataFrame, SQLContext}

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val obsDF: DataFrame = sqlContext.read.text(filePath).toDF("row").cache()

    val pipeline = PipelineBuilder.build(obsDF)

    val pipelineModel: PipelineModel = pipeline.fit(obsDF)
    val lReg = pipelineModel.stages(7).asInstanceOf[MyLinearModelImpl]
    lReg.trainingError.foreach(e => {
      println("RMSE => " + e)
    })
  }
}