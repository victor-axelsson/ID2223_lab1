package se.kth.spark.lab1.task6

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{PolynomialExpansion, RegexTokenizer, VectorSlicer}
import org.apache.spark.ml.{Pipeline, PipelineModel, linalg}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.min
import se.kth.spark.lab1.{Array2Vector, DoubleUDF, Vector2DoubleUDF}

object PipelineBuilder {

  /*
  val MAX_ITER = 50
  val REG_PARAM = 0.09

  val myLR = new LinearRegression()
    .setMaxIter(MAX_ITER)
    .setRegParam(REG_PARAM)
*/

  def build(obsDF: DataFrame): Pipeline = {

    //Step1: tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("row")
      .setOutputCol("tokens")
      .setPattern(",")

    //Step2: transform with tokenizer and show 5 rows
    val ds: DataFrame = regexTokenizer.transform(obsDF)

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
    //val firstYear:Double = v2d.agg(min("year")).collect()(0).get(0).asInstanceOf[Double]
    val firstYear:Double = 1922.0d
    //val lastYear:Double = v2d.agg(max("year")).collect()(0).get(0).asInstanceOf[Double]
    //val span = lastYear - firstYear
    println("FirstYear => " + firstYear)

    val lShifter = new DoubleUDF((f: Double) => {
      f - firstYear
    })
      .setInputCol("year")
      .setOutputCol("label")

    val normalizedDf = lShifter.transform(v2d)

    //Step7: extract just the 3 first features in a new vector column
    val fSlicer = new VectorSlicer()
      .setInputCol("fields")
      .setOutputCol("extracted_fields")
      .setIndices(Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))

    //create the polynomial expansion transformer
    val polynomialExpansionT = new PolynomialExpansion()
      .setInputCol("extracted_fields")
      .setOutputCol("features")
      .setDegree(2)

    //val featruesDf = fSlicer.transform(normalizedDf)

    val myLR = new MyLinearRegressionImpl()
    /*
    val myLR = new LinearRegression()
      .setMaxIter(50)
      .setRegParam(0.1)
      .setElasticNetParam(0.1)
    */

    val pipeline = new Pipeline().setStages(
      Array(
        regexTokenizer,
        vectorTr,
        lSlicer,
        v2dTr,
        lShifter,
        fSlicer,
        polynomialExpansionT,
        myLR
      ))

    pipeline
  }

  /*
  def getBestModel(pipeline:Pipeline, obsDF: DataFrame): LinearRegressionModel = {
    val paramGrid = new ParamGridBuilder()
      .addGrid(myLR.maxIter, Array(1, 20, 40, 60, 80, 100))
      .addGrid(myLR.regParam, Array(0.01, 0.05, 0.09, 0.15, 0.25, 0.45))
      .build()

    val cvModel: CrossValidatorModel = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .fit(obsDF)

    val lrModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages(7).asInstanceOf[LinearRegressionModel]
    lrModel
  }
  */

}
