package edu.wpi.ssogden.layercakelibrary

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.core.graphics.scale
import java.io.ByteArrayOutputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Semaphore
import java.util.concurrent.atomic.AtomicInteger
import kotlin.system.measureTimeMillis

abstract class InferenceRequest(
  context: Context,
  val creationTime: Double = Common.unixTime(),
  val minAccuracy: Double,
  val maxLatency: Double
) : Comparable<InferenceRequest> {
  companion object {
    val nextID = AtomicInteger(0)
  }

  private val TAG: String = "InferenceRequest"

  override fun toString(): String {
    return "InferenceRequest<${id}, ${submissionTime}, ${complete.count==0L}>"
  }
  var submissionTime : Double? = null
  var executionTime : Double? = null
  var completionTime : Double? = null

  val id = nextID.getAndIncrement().toString()

  val editSemaphore = Semaphore(1)
  val backupSemaphore = Semaphore(1)
  val complete = CountDownLatch(1)
  val data : Bitmap = BitmapFactory.decodeStream(context.assets.open("images/mug_224.jpg")).scale(Common.DEFAULT_MODEL_DIMENSIONS, Common.DEFAULT_MODEL_DIMENSIONS)
  val dataSize : Int =  getRemoteData().getSize() // todo: make this not happen since it takes a while, and wouldn't be known

  var modelVariantUsed : ModelVariant? = null
  var modelVariantsTried : MutableList<ModelVariant> = mutableListOf()
  var networkLatencyEstimate : Double? = null


  abstract fun getLocalData(dataSize: Int = Common.DEFAULT_MODEL_DIMENSIONS): RequestData
  abstract fun getRemoteData(dataSize: Int = Common.DEFAULT_MODEL_DIMENSIONS): RequestData

  override fun compareTo(other: InferenceRequest): Int {
    return if (Common.QUEUE_SORT_FUNC == Common.Companion.QueueSortFunctions.FIFO) {
      this.creationTime.compareTo(other.creationTime)
    } else {
      (this.creationTime + this.maxLatency).compareTo(other.creationTime + other.maxLatency)
    }
  }

  fun markSubmission() {
    submissionTime = Common.unixTime()
  }

  fun markExecution() {
    executionTime = Common.unixTime()
    // we can calculate actual time budget as this time to the deadline
  }

  fun markComplete() {
    completionTime = Common.unixTime()
    complete.countDown()
  }

  fun getResponseTime() : Double { return (completionTime!! - creationTime) }
  fun getQueueTime() : Double {return (executionTime!! - submissionTime!!)}
  fun getExecutionTime() : Double {return (completionTime!! - executionTime!!)}

  fun isComplete(): Boolean {
    return 0L == complete.count
  }

  fun getCompleteInformation(): Map<String, Any?> {
    return mapOf(
      "id" to id,
      "dataSize" to dataSize,
      "creationTime" to creationTime,
      "submissionTime" to submissionTime,
      "executionTime" to executionTime,
      "completionTime" to completionTime,
      "minAccuracy" to minAccuracy,
      "maxLatency" to maxLatency,

      "responseTime" to getResponseTime(),
      "queueTime" to getQueueTime(),
      "executionTime" to getExecutionTime(),

      "modelVariantsTried" to modelVariantsTried.map { m -> m.name },
      "modelVariant" to modelVariantUsed!!.name,

      "networkLatencyEstimate" to networkLatencyEstimate,

      "completedWithinLatency" to ((submissionTime!! + maxLatency) >= completionTime!!),
      "accuracy" to modelVariantUsed!!.accuracy
    )

  }

}



class ImageRequest(
  context: Context,
  creationTime: Double = Common.unixTime(),
  minAccuracy: Double,
  maxLatency: Double
) : InferenceRequest(context, creationTime, minAccuracy, maxLatency) {

  private val TAG: String = "ImageRequest"

  override fun getLocalData(dataSize: Int): RequestData {
    return ImageData(getResizedBitmap(dataSize))
  }

  override fun getRemoteData(dataSize: Int): RequestData {
    return ImageData(getResizedJpegByteArray(dataSize))
  }


  fun getResizedBitmap(size: Int = Common.DEFAULT_MODEL_DIMENSIONS) : Bitmap {
    return data.scale(size, size)
  }

  fun getResizedJpegByteArray(size: Int = Common.DEFAULT_MODEL_DIMENSIONS) : ByteArray {
    val baos = ByteArrayOutputStream()
    //Log.d(TAG, "resize data: ${
      measureTimeMillis {
        getResizedBitmap(size).compress(
          Bitmap.CompressFormat.JPEG,
          Common.JPEG_QUALITY,
          baos
        ) // bm is the bitmap object
      }
    //} (${size}x${size})")
    return baos.toByteArray()
  }
}


class TextRequest (
  context: Context,
  creationTime: Double = Common.unixTime(),
  minAccuracy: Double,
  maxLatency: Double
) : InferenceRequest(context, creationTime, minAccuracy, maxLatency) {

  override fun getLocalData(dataSize: Int): RequestData {
    return TextData(hashMapOf(
      "query" to "What is the meaning of life?",
      "context" to "The number of roads that a man must walk down before he can call himself human"
    ))
  }

  override fun getRemoteData(dataSize: Int): RequestData {
    return getLocalData()
  }
}


abstract class RequestData(
  open val requestData: Any
) {
  abstract fun getSize() : Int
  abstract fun getData() : Any
}

class ImageData(
  override val requestData: ByteArray
) : RequestData(requestData) {
  constructor(requestData: Bitmap) : this(ByteArray(0))

  override fun getData(): ByteArray {
    return requestData as ByteArray
  }

  override fun getSize(): Int {
    return (requestData as ByteArray).size
  }
}

class TextData(
  override val requestData: HashMap<String, String>
) : RequestData(requestData) {
  override fun getData(): HashMap<String, String> {
    return requestData as HashMap<String, String>
  }

  override fun getSize(): Int {
    var size = 0
    for (value : String in requestData.values) {
      size += value.length
    }
    return size
  }
}