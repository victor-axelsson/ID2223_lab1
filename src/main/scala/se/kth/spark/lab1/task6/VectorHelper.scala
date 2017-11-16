package se.kth.spark.lab1.task6

import org.apache.spark.ml.linalg.{DenseVector, Matrices, Vector, Vectors}

object VectorHelper {
  def dot(v1: Vector, v2: Vector): Double = {
    if(v1.size != v2.size){
      println(v1.size)
      println(v2.size)
      throw new IllegalArgumentException("The dot product of two vector means that they have to be of same size")
    }

    var sum:Double = 0
    for( i <- 0 to v1.size -1){
      sum += v1(i) * v2(i)
    }

    sum
  }

  def dot(v: Vector, s: Double): Vector = {
    var sum:Double = 0

    var newVals:Array[Double] = new Array(v.size)
    for( i <- 0 to v.size -1){
      newVals(i) = s * v(i)
    }

    new DenseVector(newVals)
  }

  def sum(v1: Vector, v2: Vector): Vector = {
    if(v1.size != v2.size){
      throw new IllegalArgumentException("The sum of two vector means that they have to be of same size")
    }

    var newVals:Array[Double] = new Array(v1.size)
    for( i <- 0 to v1.size -1){
      newVals(i) = v1(i) + v2(i)
    }

    new DenseVector(newVals)
  }

  def fill(size: Int, fillVal: Double): Vector = {
    var newVals:Array[Double] = new Array(size)
    for( i <- 0 to size -1){
      newVals(i) = fillVal
    }

    new DenseVector(newVals)
  }
}