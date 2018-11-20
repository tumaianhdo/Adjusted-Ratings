package scala

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import util.control.Breaks._
import java.io._

object CollabFilt {

  //mapping task
  def parseRating(str: String): String = {
    val fields = str.split(",")
    assert(fields.size == 3)
    val field0 = fields(0).hashCode()
    val field1 = fields(1).hashCode()
    val field2 = (fields(2).charAt(0).toDouble-48.0)
    val finalStr =field0.toString + "," + field1.toString + "," + field2.toString
    finalStr
  }

  def main(args: Array[String]): Unit = {
    //set up
    val startTimeMillis = System.currentTimeMillis()
    val conf = new SparkConf().setAppName("HW2Task1").setMaster("local[*]")
    val spark_context = new SparkContext(conf)
    val sparkSession = org.apache.spark.sql.SparkSession.builder
      .config(conf = conf)
      .appName("collab filt")
      .getOrCreate()

    val df = sparkSession.read.option("multiLine", true).json(args(0))
    df.createOrReplaceTempView("train")
    val newDF = sparkSession.sql("SELECT user_id, business_id, stars FROM train")
    newDF.show()
    val train_temp = newDF.rdd
    val train_temp1 = train_temp.map(row=> parseRating(row.toString().substring(1,row.toString().length-1)))

    val train_temp2 = train_temp1.map(_.split(',') match { case Array(user, item, rate) =>
      Rating(user.toInt, item.toInt, rate.toFloat)
    })
    val train_ratings = train_temp2.mapPartitionsWithIndex(
      (index, it) => if (index == 0) it.drop(1) else it,
      preservesPartitioning = true
    )
    //read and map TEST
    val df1 = sparkSession.read.option("multiLine", true).json(args(1))
    df1.createOrReplaceTempView("test")
    val testDF = sparkSession.sql("SELECT user_id, business_id, stars FROM test")
    testDF.show()
    val test_temp = testDF.rdd
    val test_temp1 = test_temp.map(row=> parseRating(row.toString().substring(1,row.toString().length-1)))
    val test_temp2 = test_temp1.map(_.split(',') match { case Array(user, item, rate) =>
      Rating(user.toInt, item.toInt, rate.toFloat)
    })
    val test_ratings = test_temp2.mapPartitionsWithIndex(
      (index, it) => if (index == 0) it.drop(1) else it,
      preservesPartitioning = true
    )

    // Build the recommendation model using ALS on the training data
    breakable {
      val model = ALS.train(train_ratings, 5, 5, 0.285)
      // Evaluate the model on rating data
      val usersProducts = test_ratings.map { case Rating(user, product, rate) =>
        (user, product)
      }
      val test_star_map = collection.mutable.Map[String, Double]()
      val predictions =
        model.predict(usersProducts).map { case Rating(user, product, rate) =>
          ((user, product), rate)
        }
      predictions.collect().foreach(a => {
        test_star_map += ((a._1._1.toString+","+a._1._2.toString)->a._2)
      })
      val ratesAndPreds = test_ratings.map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }.join(predictions)
      val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
        val err = (r1 - r2)
        err * err
      }.mean()
      val RMSE = Math.sqrt(MSE)

      //deal with counting error differences
      var zero_one = 0
      var one_two = 0
      var two_three = 0
      var three_four = 0
      var four_five = 0
      val erra = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
        (user,(r1 - r2))}.collect().foreach(q=>{
        if(q._2 < 1.0){
          zero_one += 1
        } else if(q._2 < 2.0){
          one_two += 1
        } else if(q._2 < 3.0){
          two_three += 1
        } else if(q._2 < 4.0){
          three_four += 1
        } else{
          four_five += 1
        }
      })
      val endTimeMillis = System.currentTimeMillis()
      println(">=0 and <1: " + zero_one)
      println(">=1 and <2: " + one_two)
      println(">=2 and <3: " + two_three)
      println(">=3 and <4: " + three_four)
      println(">=4: " + four_five)
      println("Root Mean Squared Error: " + RMSE)
      println("Time: " + (endTimeMillis - startTimeMillis)/1000)

      val pw = new PrintWriter(new File("David_Goodfellow_ModelBasedCF.txt" ))
      test_star_map.foreach(a=>{
        pw.write(a._1+","+a._2.toString+"\n")
      })
    }
  }
}
