package se.kth.spark.lab1.task6

import org.apache.spark._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

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

    val tmp = pipeline.fit(obsDF).transform(obsDF).cache()
    tmp.show(5)
    val trainingData = tmp.rdd.map((f: Row) => {
      new Instance(f.get(4).asInstanceOf[Double], f.get(7).asInstanceOf[DenseVector])
    })
    val n = tmp.count()

    //val bestmodel = PipelineBuilder.getBestModel(pipeline, obsDF)

    val numIters = 1000
    val linearRegression = new MyLinearRegressionImpl()
    var w = VectorHelper.fill(90, 0)

    val alpha = 0.5
    for (i <- 0 until numIters) {
      //compute this iterations set of predictions based on our current weights
      val labelAndPredictRDD = Helper.predict(w, trainingData)
      //compute this iteration’s RMSE

      //errorTrain(i) = Helper.rmse(labelAndPredictRDD)
      val errorTrain = Helper.rmse(labelAndPredictRDD)
      println("RMSE => " + errorTrain)

      //compute the gradient
      val gradient = linearRegression.gradient(trainingData, w)

      //update the gradient step - the alpha
      val alpha_i = alpha / (n * scala.math.sqrt(i + 1))

      //update weigths based on gradient and alpha
      val wAux = VectorHelper.dot(gradient, (-1) * alpha_i)
      w = VectorHelper.sum(w, wAux)
    }

  }
}