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
    val rawDF = sqlContext.read.text(filePath).toDF("row").cache()

    //Step1: tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("row")
      .setOutputCol("tokens")
      .setPattern(",")

    //Step2: transform with tokenizer and show 5 rows
    val ds: DataFrame = regexTokenizer.transform(rawDF)
    ds.show(5)

    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)
    val arr2Vect = new Array2Vector()
    val vectorTr = arr2Vect
      .setInputCol("tokens")
      .setOutputCol("fields")

    val vectorDF = vectorTr.transform(ds)
    vectorDF.show(5)

    //Step4: extract the label(year) into a new column
    val lSlicer = new VectorSlicer()
      .setInputCol("fields")
      .setOutputCol("yearArr")
      .setIndices(Array(0))

    val withYearDf = lSlicer.transform(vectorDF)
    withYearDf.show(5)


    //Step5: convert type of the label from vector to double (use our Vector2Double)
    val v2dTr = new Vector2DoubleUDF((f: linalg.Vector) => {
      f(0)
    })
      .setInputCol("yearArr")
      .setOutputCol("year")
    val v2d = v2dTr.transform(withYearDf)
    v2d.show(5)


    //Step6: shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF)

    val firstYear:Double = v2d.agg(min("year")).collect()(0).get(0).asInstanceOf[Double]
    val lastYear:Double = v2d.agg(max("year")).collect()(0).get(0).asInstanceOf[Double]
    val span = lastYear - firstYear

    println("First year: " + firstYear)
    println("Last year: " + lastYear)
    println("Span: " + span)


    val lShifter = new DoubleUDF((f: Double) => {
      f - firstYear
    })
      .setInputCol("year")
      .setOutputCol("label")

    val normalizedDf = lShifter.transform(v2d)
    normalizedDf.show(5)

    //Step7: extract just the 3 first features in a new vector column
    val fSlicer = new VectorSlicer()
      .setInputCol("fields")
      .setOutputCol("features")
      .setIndices(Array(1, 2, 3))

    val featruesDf = fSlicer.transform(normalizedDf)
    featruesDf.show(5)


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

    //Real year is 2001 and 1997
    val test = sqlContext.createDataFrame(Seq(
      (1, "2001.0,0.884123733793,0.610454259079,0.600498416968,0.474669212493,0.247232680947,0.357306088914,0.344136412234,0.339641227335,0.600858840135,0.425704689024,0.60491501652,0.419193351817"),
      (2, "1997.0,0.5777096107,0.589552603413,0.558605119245,0.529434582861,0.24984801906,0.482406913339,0.469210612701,0.39354097987,0.554281437919,0.487293307165,0.588379466842,0.469291436908")
    )).toDF("id", "row")


    //Step10: transform data with the model - do predictions
    val prediction = pipelineModel.transform(test)
    prediction.show(5)

    //Step11: drop all columns from the dataframe other than label and features
    val modelDf = prediction.select("features", "label")
    modelDf.show(5)

    //pipelineModel.write.overwrite().save("src/main/resources/fittedTransformationModel")

    /*
      model.write.overwrite().save("/tmp/spark-logistic-regression-model")
      pipeline.write.overwrite().save("/tmp/unfit-lr-model")
    */
  }
}