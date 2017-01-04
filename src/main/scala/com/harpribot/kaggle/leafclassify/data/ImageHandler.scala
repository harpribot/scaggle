package com.harpribot.kaggle.leafclassify.data

import scala.io.Source._
import org.imgscalr._
import java.io.File
import javax.imageio.ImageIO

/**
  * Created by harshal on 1/3/17.
  */
class ImageHandler(image_dir: String, train_csv: String, test_csv: String) {
  val imageFiles: Array[File] = recursiveListFiles(new File(image_dir))
  //val imageTestFiles: Array[File] = recursiveListFiles(new File(image_dir + "/test"))

  /**
    *
    * @param f
    * @return
    */
  private def recursiveListFiles(f: File): Array[File] = {
    val these = f.listFiles
    these ++ these.filter(_.isDirectory).flatMap(recursiveListFiles)
    val imagefiles = these.filter(!_.isDirectory)
    imagefiles
  }

  /**
    *
    */
  def createTrainImages(): Unit = {
    val idsAndLabels = fromFile(train_csv).getLines.map(_.split(",")).map(x => x(0) -> x(1)).toList.tail
    val ids: List[String] = idsAndLabels.map(_._1)
    val labels: List[String] = idsAndLabels.map(_._2)
    val trainImages: List[File] = getCorrespondingImages(ids, true)
    processImages(trainImages, labels, ids, 128, "train")
  }

  /**
    *
    */
  def createTestImages(): Unit = {
    val ids: List[String] = fromFile(test_csv).getLines.map(_.split(",")).map(x => x(0)).toList
    val labels: List[String] = ids.map(x => "")
    val testImages: List[File] = getCorrespondingImages(ids)
    processImages(testImages, labels, ids, 128, "test")
  }

  /**
    *
    * @param ids
    * @return
    */
  private def getCorrespondingImages(ids: List[String], isTrain: Boolean = true): List[File] = {
    val imageMap = imageFiles.map(x => x.getName.split('.')(0) -> x).toMap

    ids.map(x => imageMap(x))
  }

  /**
    *
    * @param img
    * @return
    */
  private def makeSquare(img: java.awt.image.BufferedImage): java.awt.image.BufferedImage = {
    val w = img.getWidth
    val h = img.getHeight
    val dim = List(w, h).min

    img match {
      case x if w == h => img
      case x if w > h => Scalr.crop(img, (w-h)/2, 0, dim, dim)
      case x if w < h => Scalr.crop(img, 0, (h-w)/2, dim, dim)
    }
  }

  /**
    *
    * @param img
    * @param width
    * @param height
    * @return
    */
  private def resizeImg(img: java.awt.image.BufferedImage, width: Int, height: Int) = {
    Scalr.resize(img, Scalr.Method.BALANCED, width, height)
  }

  /**
    *
    * @param red
    * @param green
    * @param blue
    * @return
    */
  def pixels2gray(red: Int, green:Int, blue: Int): Int = (red + green + blue) / 3

  /**
    *
    * @param img
    * @return
    */
  def image2gray(img: java.awt.image.BufferedImage): Vector[Int] = image2vec(img, pixels2gray)

  /**
    *
    * @param img
    * @param f
    * @tparam A
    * @return
    */
  private def image2vec[A](img: java.awt.image.BufferedImage, f: (Int, Int, Int) => A ): Vector[A] = {
    val w = img.getWidth
    val h = img.getHeight
    for { w1 <- (0 until w).toVector
          h1 <- (0 until h).toVector
    } yield {
      val col = img.getRGB(w1, h1)
      val red =  (col & 0xff0000) / 65536
      val green = (col & 0xff00) / 256
      val blue = col & 0xff
      f(red, green, blue)
    }
  }

  /**
    *
    * @param imgFiles
    * @param labels
    * @param ids
    * @param resizeImgDim
    * @param dataNature
    */
  private def processImages(imgFiles: List[File], labels: List[String], ids: List[String], resizeImgDim: Int = 128, dataNature: String): Unit = {
    //lazy val imageTrainFiles: Array[File] = recursiveListFiles(new File(image_dir + "/train"))
    val processedImages: List[java.awt.image.BufferedImage] =
      imageFiles.map(x => resizeImg(makeSquare(ImageIO.read(x)), resizeImgDim, resizeImgDim)).toList

    processedImages.zip(labels.zip(ids)).foreach{x =>
      val dir: File = new File(image_dir + "/" + dataNature + "/" + x._2._1)
      if (!dir.exists()) dir.mkdirs()
      val image_file: File = new File(image_dir + "/" + dataNature + "/" + x._2._1 + "/" + x._2._2 + ".jpg")
      image_file.createNewFile()
      ImageIO.write(x._1, "jpg", image_file)
    }

  }
}
