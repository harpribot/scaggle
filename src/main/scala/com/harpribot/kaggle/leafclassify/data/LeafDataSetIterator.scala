package com.harpribot.kaggle.leafclassify.data

import org.datavec.api.io.filters.BalancedPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.api.split.InputSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.transform.FlipImageTransform
import org.datavec.image.transform.ImageTransform
import org.datavec.image.transform.WarpImageTransform
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import java.io.File
import java.util.Random

import org.datavec.image.recordreader.ImageRecordReader


/**
  * Created by harshal on 1/3/17.
  */
class LeafDataSetIterator(image_dir: String,
                          train_csv: String,
                          test_csv: String,
                          rng: Random,
                          numExamples: Int,
                          numLabels: Int,
                          batchSize: Int,
                          splitTrainVal: Double,
                          height: Int,
                          width: Int,
                          channels: Int,
                          epochs: Int,
                          nCores: Int
                         ) {
  val im_handle: ImageHandler = new ImageHandler(image_dir, train_csv, test_csv)

  /**
    *
    * @param useTransform
    * @return
    */
  def getTrainAndVal(useTransform: Boolean = false): (List[MultipleEpochsIterator], DataSetIterator) = {
    im_handle.createTrainImages()
    val labelMaker: ParentPathLabelGenerator = new ParentPathLabelGenerator()
    val mainPath: File = new File(image_dir + "/" + "train/")
    val fileSplit: FileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng)
    val pathFilter: BalancedPathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, batchSize)

    val inputSplit: Array[InputSplit] = fileSplit.sample(pathFilter, numExamples * (1 + splitTrainVal), numExamples * (1 - splitTrainVal))
    val trainData: InputSplit = inputSplit(0)
    val valData: InputSplit = inputSplit(1)
    var transforms: List[ImageTransform] = null
    if (useTransform){
      transforms = getTransforms(rng)
    }

    val scaler: DataNormalization = new ImagePreProcessingScaler(0, 1)

    val recordReader: ImageRecordReader = new ImageRecordReader(height, width, channels, labelMaker)
    val trainBatchIters: List[MultipleEpochsIterator] = List[MultipleEpochsIterator]()

    if (useTransform){
      for (transform: ImageTransform <- transforms){
        recordReader.initialize(trainData, transform)
        val trainDataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels)
        scaler.fit(trainDataIter)
        trainDataIter.setPreProcessor(scaler)
        val trainBatchIter = new MultipleEpochsIterator(epochs, trainDataIter, nCores)
        trainBatchIters :+ trainBatchIter
      }
    }
    else{
      recordReader.initialize(trainData, null)
      val trainDataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels)
      scaler.fit(trainDataIter)
      val trainBatchIter = new MultipleEpochsIterator(epochs, trainDataIter, nCores)
      trainBatchIters :+ trainBatchIter
    }

    recordReader.initialize(valData)
    val valDataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels)
    scaler.fit(valDataIter)
    valDataIter.setPreProcessor(scaler)

    (trainBatchIters, valDataIter)
  }

  /**
    *
    * @param rng
    * @return
    */
  def getTransforms(rng: Random): List[ImageTransform] = {
    val flipTransform1: ImageTransform= new FlipImageTransform(rng)
    val flipTransform2: ImageTransform = new FlipImageTransform(new Random(123))
    val warpTransform: ImageTransform = new WarpImageTransform(rng, 42)

    val transforms: List[ImageTransform] = List[ImageTransform](flipTransform1, warpTransform, flipTransform2)

    transforms
  }

  /**
    *
    * @return
    */
  def getTestVal: DataSetIterator = {
    val labelMaker: ParentPathLabelGenerator = new ParentPathLabelGenerator()
    val mainPath: File = new File(System.getProperty("user.dir"), image_dir + "/" + "train/")
    val fileSplit: FileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng)
    val pathFilter: BalancedPathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, batchSize)
    val splitTest = 1
    val inputSplit: Array[InputSplit] = fileSplit.sample(pathFilter, numExamples * (1 + splitTest), numExamples * (1 - splitTest))
    val trainData: InputSplit = inputSplit(0)
    val valData: InputSplit = inputSplit(1)

    val flipTransform1: ImageTransform= new FlipImageTransform(rng)
    val flipTransform2: ImageTransform = new FlipImageTransform(new Random(123))
    val warpTransform: ImageTransform = new WarpImageTransform(rng, 42)

    val transforms: List[ImageTransform] = List[ImageTransform](flipTransform1, warpTransform, flipTransform2)

    val scaler: DataNormalization = new ImagePreProcessingScaler(0, 1)

    val recordReader: ImageRecordReader = new ImageRecordReader(height, width, channels, labelMaker)

    recordReader.initialize(trainData, null)
    val dataIter: DataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels)

    scaler.fit(dataIter)
    dataIter.setPreProcessor(scaler)

    dataIter
  }
}
