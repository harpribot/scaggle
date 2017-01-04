package com.harpribot.kaggle.leafclassify.model

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf._
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution
import org.deeplearning4j.nn.conf.distribution.NormalDistribution
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * Created by harshal on 1/4/17.
  */
class AlexNet(val seed: Long,
              val iterations: Int,
              val channels: Int,
              val numLabels: Int,
              val height: Int,
              val width: Int) extends ConvNet{
  val nonZeroBias = 1
  val dropOut = 0.5
  val model: MultiLayerNetwork = createModel


  def createModel: MultiLayerNetwork = {
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .weightInit(WeightInit.DISTRIBUTION)
      .dist(new NormalDistribution(0.0, 0.01))
      .activation(Activation.RELU)
      .updater(Updater.NESTEROVS)
      .iterations(iterations)
      .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(1e-4)
      .biasLearningRate(1e-2*2)
      .learningRateDecayPolicy(LearningRatePolicy.Step)
      .lrPolicyDecayRate(0.01)
      .lrPolicySteps(100000)
      .regularization(true)
      .l2(5 * 1e-4)
      .momentum(0.9)
      .miniBatch(false)
      .list()
      .layer(0, convInit("cnn1", channels, 96, Array(11, 11), Array(4, 4), Array(3, 3), 0))
      .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
      .layer(2, maxPool("maxpool1", Array(3,3)))
      .layer(3, conv5x5("cnn2", 256, Array(1,1), Array(2,2), nonZeroBias))
      .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
      .layer(5, maxPool("maxpool2", Array(3,3)))
      .layer(6,conv3x3("cnn3", 384, 0))
      .layer(7,conv3x3("cnn4", 384, nonZeroBias))
      .layer(8,conv3x3("cnn5", 256, nonZeroBias))
      .layer(9, maxPool("maxpool3", Array(3,3)))
      .layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
      .layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
      .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .name("output")
        .nOut(numLabels)
        .activation(Activation.SOFTMAX)
        .build())
      .backprop(true)
      .pretrain(false)
      .setInputType(InputType.convolutional(height, width, channels))
      .build()

    new MultiLayerNetwork(conf)
  }

  def getModel: MultiLayerNetwork = {
    model
  }

}
