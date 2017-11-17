package se.kth.spark.lab1.task2


import se.kth.spark.lab1._
import org.apache.spark.ml.feature.{RegexTokenizer, VectorSlicer}
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.{Pipeline, linalg}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import se.kth.spark.lab1.Vector2DoubleUDF
import org.apache.spark.sql.functions.{min, max}

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val splits = sqlContext.read.text(filePath).toDF("row").randomSplit(Array(0.7, 0.3))

    val rawDF = splits(0)
    val testDF = splits(1)

    //Step1: tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("row")
      .setOutputCol("tokens")
      .setPattern(",")

    //Step2: transform with tokenizer and show 5 rows
    val ds: DataFrame = regexTokenizer.transform(rawDF)

    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)
    val arr2Vect = new Array2Vector()
    val vectorTr = arr2Vect
      .setInputCol("tokens")
      .setOutputCol("fields")

    val vectorDF = vectorTr.transform(ds)

    //Step4: extract the label(year) into a new column
    val lSlicer = new VectorSlicer()
      .setInputCol("fields")
      .setOutputCol("yearArr")
      .setIndices(Array(0))

    val withYearDf = lSlicer.transform(vectorDF)

    //Step5: convert type of the label from vector to double (use our Vector2Double)
    val v2dTr = new Vector2DoubleUDF((f: linalg.Vector) => {
      f(0)
    })
      .setInputCol("yearArr")
      .setOutputCol("year")
    val v2d = v2dTr.transform(withYearDf)

    //Step6: shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF)
    val firstYear:Double = v2d.agg(min("year")).collect()(0).get(0).asInstanceOf[Double]
    val lastYear:Double = v2d.agg(max("year")).collect()(0).get(0).asInstanceOf[Double]
    val span = lastYear - firstYear

    println("First year => " + firstYear)
    println("Last year => " + lastYear)
    println("Span => " + span)


    val lShifter = new DoubleUDF((f: Double) => {
      f - firstYear
    })
      .setInputCol("year")
      .setOutputCol("label")

    val normalizedDf = lShifter.transform(v2d)

    //Step7: extract just the 3 first features in a new vector column
    val fSlicer = new VectorSlicer()
      .setInputCol("fields")
      .setOutputCol("features")
      .setIndices(Array(1, 2, 3))

    val featruesDf = fSlicer.transform(normalizedDf)

    //Step8: put everything together in a pipeline
    val pipeline = new Pipeline().setStages(
      Array(
        regexTokenizer,
        vectorTr,
        lSlicer,
        v2dTr,
        lShifter,
        fSlicer
      ))

    //Step9: generate model by fitting the rawDf into the pipeline
    val pipelineModel = pipeline.fit(rawDF)

    //Step10: transform data with the model - do predictions
    val prediction = pipelineModel.transform(testDF)
    prediction.show(5)

    //Step11: drop all columns from the dataframe other than label and features
    val modelDf = prediction.select("features", "label")
    modelDf.show(5)

  }
}