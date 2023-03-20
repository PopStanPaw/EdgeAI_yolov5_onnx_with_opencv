// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.*
import android.content.Context
import android.graphics.Bitmap
import android.graphics.ColorSpace
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.opencv.core.*
import org.opencv.core.CvType.CV_8UC3
import org.opencv.core.CvType.CV_8UC4
import org.opencv.dnn.Net
import java.io.File
import java.io.IOException
import java.lang.reflect.Type
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.text.DecimalFormat
import java.util.*
import kotlin.jvm.internal.Ref
import kotlin.math.exp
import kotlin.math.max

import org.bytedeco.javacpp.*
import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.Module
import org.bytedeco.pytorch.global.torch.*
import org.opencv.core.Scalar

internal data class Result(
        var detectedIndices: List<Int> = emptyList(),
        var detectedScore: MutableList<Float> = mutableListOf<Float>(),
        var processTimeMs: Long = 0
) {}

internal class ORTAnalyzer(
        private val ortSession: OrtSession?,
        private val callBack: (Result) -> Unit,
        private val context: Context
) : ImageAnalysis.Analyzer {

    // Get index of top 3 values
    // This is for demo purpose only, there are more efficient algorithms for topK problems
    private fun getTop3(labelVals: FloatArray): List<Int> {
        var indices = mutableListOf<Int>()
        for (k in 0..2) {
            var max: Float = 0.0f
            var idx: Int = 0
            for (i in 0..labelVals.size - 1) {
                val label_val = labelVals[i]
                if (label_val > max && !indices.contains(i)) {
                    max = label_val
                    idx = i
                }
            }

            indices.add(idx)
        }

        return indices.toList()
    }

    // Calculate the SoftMax for the input array
    private fun softMax(modelResult: FloatArray): FloatArray {
        val labelVals = modelResult.copyOf()

        if (labelVals.isEmpty()) throw NoSuchElementException()
        var max = labelVals[0]
        var lastindex = labelVals.size
        for (i in 1..lastindex) {
            val e = labelVals[i]
            max = maxOf(max, e)
        }

        var sum = 0.0f

        // Get the reduced sum
        for (i in labelVals.indices) {
            labelVals[i] = exp(labelVals[i] - max!!)
            sum += labelVals[i]
        }

        if (sum != 0.0f) {
            for (i in labelVals.indices) {
                labelVals[i] /= sum
            }
        }

        return labelVals
    }

    // Rotate the image of the input bitmap
    fun Bitmap.rotate(degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }

    @Throws(IOException::class)
    fun getWtiModelFromAssets(context: Context): File = File(context.cacheDir, "wti.onnx")
        .also {
            it.outputStream().use { cache -> context.assets.open("wti.onnx").use { it.copyTo(cache) } }
        }

    @Throws(IOException::class)
    fun getPostWtiModelFromAssets(context: Context): File = File(context.cacheDir, "postWTIfaceDetect.caffemodel")
        .also {
            it.outputStream().use { cache -> context.assets.open("postWTIfaceDetect.caffemodel").use { it.copyTo(cache) } }
        }

    @Throws(IOException::class)
    fun getPostWtiConfigFromAssets(context: Context): File = File(context.cacheDir, "postWTIconfig.prototxt")
        .also {
            it.outputStream().use { cache -> context.assets.open("postWTIconfig.prototxt").use { it.copyTo(cache) } }
        }

    fun postWtiCheckFace(inputImg: org.opencv.core.Mat){
        val scaleFactor = 1.0

        val inputImgWidth = inputImg.width()
        val inputImgHeight = inputImg.height()

        val aiImgWidth = 640.0
        val aiImgHeight = 640.0

        val inputBlob = org.opencv.dnn.Dnn.blobFromImage(inputImg, scaleFactor, Size(aiImgWidth, aiImgHeight),  Scalar(104.0, 117.0, 123.0), false, false)

        val modelFilePath = getPostWtiModelFromAssets(context).absolutePath
        val configFilePath = getPostWtiConfigFromAssets(context).absolutePath
        val net = org.opencv.dnn.Dnn.readNetFromCaffe(configFilePath, modelFilePath)

        net.setPreferableBackend(org.opencv.dnn.Dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(org.opencv.dnn.Dnn.DNN_TARGET_CPU)

        net.setInput(inputBlob)
        val confidenceThreshold = 0.7
        var bboxNum = 0.0
        var detections = Mat()
        detections = net.forward()

        for (i in 0 until 200) {
            var outputIndex = IntArray(4)
            outputIndex[0] = 0
            outputIndex[1] = 0
            outputIndex[2] = i
            outputIndex[3] = 2
            val confidence = detections.get(outputIndex)
            if (confidence[0].toDouble() > confidenceThreshold) {
                outputIndex[3] = 3
                val x1 = detections.get(outputIndex)[0].toDouble() * inputImgWidth
                outputIndex[3] = 4
                val y1 = detections.get(outputIndex)[0].toDouble() * inputImgHeight
                outputIndex[3] = 5
                val x2 = detections.get(outputIndex)[0].toDouble() * inputImgWidth
                outputIndex[3] = 6
                val y2 = detections.get(outputIndex)[0].toDouble() * inputImgHeight

                bboxNum += 1.0
            }
        }

        if (bboxNum == 0.0) {
            Log.println(Log.VERBOSE, "Detection Result", "NO PROPER FACE DETECTED.")
        }
        else {
            Log.println(Log.VERBOSE, "Detection Result", "Proper face detected.")
        }
    }

    //fun cfaROIAI(inputImg: org.opencv.core.Mat) {

    //}

    override fun analyze(image: ImageProxy) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME)

        // Shu Li, prepare input image for WTI (yolov5.onnx) AI model.
        // (-- 1 --) Convert Kotlin/Java YUV-420-888 Image to OpenCV RGB Mat Image.
        // resize to 640*480.
        val yoloBitmap = image.toBitmap()
        val rawYoloBitmap = yoloBitmap?.let { Bitmap.createScaledBitmap(it, 640, 480, false) }
        val finalBitmap = rawYoloBitmap?.rotate(image.imageInfo.rotationDegrees.toFloat())
        val inputImg = Mat()
        org.opencv.android.Utils.bitmapToMat(finalBitmap, inputImg)
        org.opencv.imgproc.Imgproc.cvtColor(inputImg, inputImg, org.opencv.imgproc.Imgproc.COLOR_RGB2BGR)

        // Validate post WTI with AI for face detection.
        postWtiCheckFace(inputImg)

        var pixelRGB = inputImg.get(320, 320).toList()
        //Log.println(Log.VERBOSE, "---------- Log Image Info. ----------", "")
        //Log.println(Log.VERBOSE, "image type", image.format.toString())
        //Log.println(Log.VERBOSE, "image Width", image.width.toString())
        //Log.println(Log.VERBOSE, "image Height", image.height.toString())
        //Log.println(Log.VERBOSE, "Image", image.planes[0].buffer.remaining().toString())
        //Log.println(Log.VERBOSE, "myImage_cols", inputImg.cols().toString())
        //Log.println(Log.VERBOSE, "myImage_rows", inputImg.rows().toString())
        //Log.println(Log.VERBOSE, "myImage_channels", inputImg.channels().toString())
        //Log.println(Log.VERBOSE, "myImage_pixel B", (pixelRGB[0]).toString())
        //Log.println(Log.VERBOSE, "myImage_pixel G", (pixelRGB[1]).toString())
        //Log.println(Log.VERBOSE, "myImage_pixel R", (pixelRGB[2]).toString())

        // (-- 2 --) Prepare blob from input image for yolov5 input.
        // Transfer image to yolov5 format (square sized).
        val imgWidth = inputImg.width()
        val imgHeight = inputImg.height()
        val targetWidth = max(imgWidth, imgHeight)
        val yoloImg: Mat = org.opencv.core.Mat.zeros(targetWidth, targetWidth, CV_8UC3)
        inputImg.copyTo(yoloImg.colRange(org.opencv.core.Range(0, imgWidth)))

        // Create blob image.
        val scaleFactor = 1.0 / 255.0
        val inputWidthYolo = 640.0
        val inputHeightYolo = 640.0
        val inputBlob = org.opencv.dnn.Dnn.blobFromImage(yoloImg, scaleFactor, org.opencv.core.Size(inputWidthYolo, inputHeightYolo), org.opencv.core.Scalar(0.0, 0.0, 0.0), true, false) as Mat

        // (-- 3 --) Provide input blob to yolov5, detect and get output.
        val yoloStartTime = SystemClock.uptimeMillis()

        val filePath =  getWtiModelFromAssets(context).absolutePath
        val net = org.opencv.dnn.Dnn.readNet(filePath)

        org.opencv.imgcodecs.Imgcodecs.imwrite(context.cacheDir.absolutePath + SystemClock.uptimeMillis().toString() + "_yolo.jpg", yoloImg)
        org.opencv.imgcodecs.Imgcodecs.imwrite(context.cacheDir.absolutePath + SystemClock.uptimeMillis().toString() + "_input.jpg", inputImg)
        //Log.println(Log.VERBOSE, "Filename =", context.cacheDir.absolutePath + "_input.jpg")

        net.setPreferableBackend(org.opencv.dnn.Dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(org.opencv.dnn.Dnn.DNN_TARGET_CPU)

        net.setInput(inputBlob)
        var yoloOutputs = arrayListOf<Mat>()
        net.forward(yoloOutputs, net.unconnectedOutLayersNames)

        // (-- 4 --) Process yolov5 outputs and obtain labels, confidence, and set process time.
        val rows = 25200
        val pageIndex = 0
        val confidenceThreshold = 0.2.toFloat()
        val nmsThreshold = 0.4.toFloat()
        val scoreThreshold = 0.2.toFloat()
        val xFactor = (yoloImg.cols() / inputWidthYolo).toFloat()
        val yFactor = (yoloImg.rows() / inputHeightYolo).toFloat()

        var classIds = arrayListOf<Int>()
        var confidences = arrayListOf<Float>()
        var processingTimes = arrayListOf<Float>()
        var bBoxes: MutableList<Rect2d> = mutableListOf()
        Log.println(Log.VERBOSE, "bBoxes initial size", bBoxes.size.toString())

        val decimal = DecimalFormat("#.###########")

        for (i in 0 until rows){
            var outputIndex = IntArray(3)
            outputIndex[0] = pageIndex
            outputIndex[1] = i
            outputIndex[2] = 4
            //val confidenceIndex = yoloOutputs[0].at(Float::class.java, outputIndex)
            val confidence = yoloOutputs[0].get(outputIndex)
            //Log.println(Log.VERBOSE, "Confidence Size", confidence.size.toString())
            //Log.println(Log.VERBOSE, "Confidecne Value", decimalConfi.format(confidence[0].toFloat()).toString())

            if (confidence[0].toFloat() >= confidenceThreshold) {
                // Find class ID in the trained labels.
                var classesScores = FloatArray(6)
                for (j in 0..5) {
                    outputIndex[2] = outputIndex[2] + 1
                    classesScores[j] = (yoloOutputs[0].get(outputIndex))[0].toFloat()
                }
                val maxScore = (classesScores.toList().maxOrNull() ?: 0).toFloat()
                Log.println(Log.VERBOSE, "Max Score", decimal.format(maxScore).toString())

                if (maxScore > scoreThreshold) {
                    confidences.add(confidence[0].toFloat())
                    classIds.add(classesScores.toList().indexOf(maxScore))
                    Log.println(Log.VERBOSE, "detected class label ID", (classesScores.toList().indexOf(maxScore)).toString())

                    //yoloOutputs[0].at<Mat.Atable<Float>>(outputIndex)

                    outputIndex[2] = 0
                    val x = (yoloOutputs[0].get(outputIndex))[0].toFloat()
                    outputIndex[2] = 1
                    val y = (yoloOutputs[0].get(outputIndex))[0].toFloat()
                    outputIndex[2] = 2
                    val w = (yoloOutputs[0].get(outputIndex))[0].toFloat()
                    outputIndex[2] = 3
                    val h = (yoloOutputs[0].get(outputIndex))[0].toFloat()

                    //Log.println(Log.VERBOSE, "---------- Log bounding box and box scaling. ----------", "")
                    //Log.println(Log.VERBOSE, "box x", decimal.format(x))
                    //Log.println(Log.VERBOSE, "box y", decimal.format(y))
                    //Log.println(Log.VERBOSE, "box w", decimal.format(w))
                    //Log.println(Log.VERBOSE, "box h", decimal.format(h))

                    val left = (((x - 0.5 * w) * xFactor).toInt())
                    val top = (((y - 0.5 * h) * yFactor).toInt())
                    val width = ((w * xFactor).toInt())
                    val height = ((h * yFactor).toInt())
                    //Log.println(Log.VERBOSE, "box left", decimal.format(left))
                    //Log.println(Log.VERBOSE, "box top", decimal.format(top))
                    //Log.println(Log.VERBOSE, "box width", decimal.format(width))
                    //Log.println(Log.VERBOSE, "box height", decimal.format(height))

                    val rectangle = Rect2d(left.toDouble(), top.toDouble(), width.toDouble(), height.toDouble())

                    bBoxes.add(rectangle)
                }
            }
        }

        // Prepare for NMS (non-maximum supression).
        val confidencesCV: MatOfFloat = MatOfFloat()
        confidencesCV.fromList(confidences.toList())
        val bBoxesCV: MatOfRect2d = MatOfRect2d()
        bBoxesCV.fromList(bBoxes)

        Log.println(Log.VERBOSE, "---------- Log box & confidence size. ----------", "")
        Log.println(Log.VERBOSE, "confidences size", confidencesCV.size().toString())
        Log.println(Log.VERBOSE, "bbox size", bBoxesCV.size().toString())

        var nmsResultCV = MatOfInt()
        org.opencv.dnn.Dnn.NMSBoxes(bBoxesCV, confidencesCV, scoreThreshold, nmsThreshold, nmsResultCV)

        Log.println(Log.VERBOSE, "nms Width", nmsResultCV.width().toString())
        Log.println(Log.VERBOSE, "nms Height", nmsResultCV.height().toString())
        Log.println(Log.VERBOSE, "nms Channel", nmsResultCV.channels().toString())

        // Filter final detection results.
        var yoloResIndices = mutableListOf<Int>()
        var yoloResConfidences = mutableListOf<Float>()
        if (nmsResultCV.width() >= 1 && nmsResultCV.height() >= 1){
            val nmsResult = nmsResultCV.toArray()
            var controlKey = 3
            for (i in nmsResult.indices){
                if(controlKey > 0){
                    val idx = nmsResult[i]
                    yoloResIndices.add(classIds[idx])
                    yoloResConfidences.add(confidences[idx])
                    controlKey -= - 1
                }
            }
            Log.println(Log.VERBOSE, "final kept indices size", yoloResIndices.size.toString())


        }
        // if nothing is detected.
        else if(nmsResultCV.width() == 1 && nmsResultCV.height() == 0){
            yoloResIndices.add(0)
            val defaultScore: Float = (-1.0).toFloat()
            yoloResConfidences.add(defaultScore)
        }

        var yoloResult = Result()
        yoloResult.processTimeMs = SystemClock.uptimeMillis() - yoloStartTime
        yoloResult.detectedIndices = yoloResIndices.toList()
        yoloResult.detectedScore = yoloResConfidences


        /*
        // -----------------------------------------------------------------------------
        //
        //                     From Original Demo APP in Git Repo.
        //
        // -----------------------------------------------------------------------------
        // Convert the input image to bitmap and resize to 224x224 for model input
        val imgBitmap = image.toBitmap()
        val rawBitmap = imgBitmap?.let { Bitmap.createScaledBitmap(it, 224, 224, false) }
        val bitmap = rawBitmap?.rotate(image.imageInfo.rotationDegrees.toFloat())

        if (bitmap != null) {
            var result = Result()

            val imgData = preProcess(bitmap)
            val inputName = ortSession?.inputNames?.iterator()?.next()
            val shape = longArrayOf(1, 3, 224, 224)
            val env = OrtEnvironment.getEnvironment()
            env.use {
                val tensor = OnnxTensor.createTensor(env, imgData, shape)
                val startTime = SystemClock.uptimeMillis()
                tensor.use {
                    val output = ortSession?.run(Collections.singletonMap(inputName, tensor))
                    output.use {
                        result.processTimeMs = SystemClock.uptimeMillis() - startTime
                        @Suppress("UNCHECKED_CAST")
                        val rawOutput = ((output?.get(0)?.value) as Array<FloatArray>)[0]
                        val probabilities = softMax(rawOutput)
                        result.detectedIndices = getTop3(probabilities)
                        for (idx in result.detectedIndices) {
                            result.detectedScore.add(probabilities[idx])
                        }
                    }
                }
            }
            callBack(result)
        }
        */
        callBack(yoloResult)
        image.close()
    }

    // We can switch analyzer in the app, need to make sure the native resources are freed
    protected fun finalize() {
        ortSession?.close()
    }
}
