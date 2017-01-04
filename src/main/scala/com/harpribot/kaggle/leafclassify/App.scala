package com.harpribot.kaggle.leafclassify

import com.harpribot.kaggle.leafclassify.data.LeafDataSetIterator
import java.util.Random

import org.deeplearning4j.api.storage.StatsStorage
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.deeplearning4j.ui.stats.StatsListener
import org.apache.commons.io.FilenameUtils
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.deeplearning4j.util.ModelSerializer
import org.slf4j.Logger
import org.slf4j.LoggerFactory

import com.harpribot.kaggle.leafclassify.model._


import java.io.File

/**
 * @author ${user.name}
 */
object App {
  lazy val log: Logger = LoggerFactory.getLogger(App.getClass)
  def main(args : Array[String]) {
    //Nd4j.dtype = DataBuffer.Type.DOUBLE
    //Nd4j.factory().setDType(DataBuffer.Type.DOUBLE)
    //Nd4j.ENFORCE_NUMERICAL_STABILITY = true

    // parameters
    val height = 128
    val width = 128
    val channels = 1
    val numExamples = 1584
    val numLabels = 99
    val batchSize = 64
    val image_dir = "/Users/harshal/Documents/kaggle/leafclassify/data/images"
    val train_csv = "/Users/harshal/Documents/kaggle/leafclassify/data/train.csv"
    val test_csv = "/Users/harshal/Documents/kaggle/leafclassify/data/test.csv"

    val seed: Long = 42
    val rng: Random = new Random(seed)
    val listenerFreq: Int = 1
    val iterations: Int = 10
    val epochs: Int = 100
    val splitTrainVal: Double = 0.8
    val nCores: Int = 4
    val save: Boolean = false


    // UI
    //val uiServer: UIServer = UIServer.getInstance()
    //val statsStorage: StatsStorage = new InMemoryStatsStorage()
    //uiServer.attach(statsStorage)

    // model Type
    val modelType: String = "AlexNet"

    log.info("Load data....")
    val dataHandler = new LeafDataSetIterator(image_dir, train_csv, test_csv, rng,
      numExamples, numLabels, batchSize, splitTrainVal, height, width, channels, epochs, nCores)

    log.info("Build model....")
    val network: MultiLayerNetwork = new AlexNet(seed, iterations, channels, numLabels, height, width).getModel

    network.init()
    network.setListeners(new ScoreIterationListener(listenerFreq))
    //network.setListeners(new StatsListener(statsStorage))

    var trainAndVal: (List[MultipleEpochsIterator], DataSetIterator) = null


    log.info("Train model....")
    val useTransform = true
    if (useTransform) trainAndVal = dataHandler.getTrainAndVal(useTransform = true)
    else trainAndVal = dataHandler.getTrainAndVal(useTransform = false)

    val trains = trainAndVal._1
    for (train <- trains){
      network.fit(train)
    }

    log.info("Evaluate model....")
    val validate: DataSetIterator = trainAndVal._2
    val eval: Evaluation = network.evaluate(validate)
    log.info(eval.stats(true))

    log.info("Make Predictions...")


    if (save) {
      log.info("Save model....")
      val basePath: String = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/")
      val locationToSave: File = new File(basePath + "/" + "MyMultiLayerNetwork.zip")
      val saveUpdater: Boolean = true
      ModelSerializer.writeModel(network,locationToSave, saveUpdater)
    }
  }

}
