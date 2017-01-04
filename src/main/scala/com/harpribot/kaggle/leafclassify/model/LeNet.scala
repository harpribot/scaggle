package com.harpribot.kaggle.leafclassify.model

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf._
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * Created by harshal on 1/4/17.
  */
class LeNet(val seed: Long,
              val iterations: Int,
              val channels: Int,
              val numLabels: Int,
              val height: Int,
              val width: Int) extends ConvNet{

  val model: MultiLayerNetwork = createModel


  def createModel: MultiLayerNetwork = {
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .regularization(false).l2(0.005) // tried 0.0001, 0.0005
      .activation(Activation.RELU)
      .learningRate(0.0001) // tried 0.00001, 0.00005, 0.000001
      .weightInit(WeightInit.XAVIER)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.RMSPROP).momentum(0.9)
      .list()
      .layer(0, convInit("cnn1", channels, 50 ,  Array(5, 5), Array(1, 1), Array(0, 0), 0))
      .layer(1, maxPool("maxpool1", Array(2,2)))
      .layer(2, conv5x5("cnn2", 100, Array(5, 5), Array(1, 1), 0))
      .layer(3, maxPool("maxool2", Array(2,2)))
      .layer(4, new DenseLayer.Builder().nOut(500).build())
      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(numLabels)
        .activation(Activation.SOFTMAX)
        .build())
      .backprop(true).pretrain(false)
      .setInputType(InputType.convolutional(height, width, channels))
      .build()

    new MultiLayerNetwork(conf)
  }

  def getModel: MultiLayerNetwork = {
    model
  }

}
