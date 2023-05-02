package edu.wpi.ssogden.layercakelibrary

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.core.graphics.scale
import org.apache.commons.math3.stat.descriptive.rank.PSquarePercentile
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.util.*
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Semaphore
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.floor
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt
import kotlin.system.measureTimeMillis

class Common {

  companion object {

    enum class QueueSortFunctions {
      FIFO,
      EDF
    }

    enum class Application {
      IMAGE,
      TEXT
    }

    val REMOTE_SERVER_IP = "34.200.230.25"

    val RUN_SEQUENTIAL = false
    val NUM_JOBS_TO_RUN = 1000
    val ARRIVAL_RATE = 1.0 // Queries per second

    val MEASURE_MODELS = true
    val NUM_MEASUREMENT_THREADS = 3
    val PERCENTILE_OF_LATENCY = 0.9
    val LATENCY_BUFFER = 0.8 // Lower means a tighter time budget is calculated

    val QUEUE_SORT_FUNC = QueueSortFunctions.EDF
    val ONLY_USE_LOCAL = true
    val ONLY_USE_REMOTE = false
    val USE_INFAAS = false

    val MIN_LATENCY = 0.25
    val PIVOT_LATENCY = 1.0 // latency which we swap around at
    val MAX_LATENCY = 3.5

    val DO_LATENCY_PIVOT = false // swap latency part way through the run

    val NUM_MODEL_MEASUREMENTS = 100

    val USE_BASELINE_MODEL = true

    val TEXT_PROBABILITY = 0.0


    val JPEG_QUALITY = 90
    val RESIZE_FOR_NETWORK = true
    val DEFAULT_MODEL_DIMENSIONS = 600 //1200
    // next try turning off resize


    val rand_entry: Random = Random(0) // Can set seed
    val rand_latency: Random = Random(1) // Can set seed
    val rand_bert: Random = Random(2) // Can set seed
    val rand_do_text: Random = Random(3) // Can set seed

    val BILLION = 1_000_000_000.0
    val MAX_CONCURRENCY = 2
    val RETRY_SLEEP_DURATION = 100L

    val MODEL_CACHE_LIFETIME = 10
    val NUM_BYTES_FOR_BANDWIDTH_ESTIMATE = 100_000

    val RESIZE_BUFFER = 50 / 1000.0 // buffer in ms


    //val MAX_CONCURRENT_JOBS = 10
    val NUM_BURNIN_REQUESTS = 20

    val EMA_ALPHA = 0.5

    val LAUNCH_BACKUP = false
    val LAUNCH_LOCAL_BACKUP = false
    val DO_PROACTIVE_BACKUP = false

    val TIME_TO_WARM_MODEL = 10.0
    val REMOTE_TIMEOUT = 5L // todo: maybe tune?



    fun percentile(list: List<Double>, percentile: Double): Double {
      return list.sortedBy { it }[floor(percentile * list.size).roundToInt()]
    }

    fun mean(list: List<Double>): Double {
      if (list.size == 0) {
        return 0.0;
      }
      return list.average()
    }

    fun stddev(list: List<Double>): Double {
      val avg = mean(list)
      return sqrt(list.map { d -> (avg - d).pow(2) }.sum() / list.size)
    }

    fun loadImage(context: Context, width: Int = 224, height: Int = 224): ByteArray {
      val img_path = "images/mug_224.jpg"
      val inputStream: InputStream = context.assets.open(img_path)
      val raw_bitmap: Bitmap = BitmapFactory.decodeStream(inputStream)
      val bm = raw_bitmap.scale(width, height)

      val baos = ByteArrayOutputStream()
      bm.compress(Bitmap.CompressFormat.JPEG, 70, baos) // bm is the bitmap object

      val b: ByteArray = baos.toByteArray()
      return b
    }

    fun unixTime(): Double {
      return System.currentTimeMillis() / 1000.0
    }

    fun getExpoNext(rate: Double): Double {
      return Math.log(1 - rand_entry.nextDouble()) / -rate
    }

    fun getRandomDouble(minVal: Double = MIN_LATENCY, maxVal: Double = MAX_LATENCY): Double {
      return rand_latency.nextDouble() * (maxVal - minVal) + minVal
    }

    fun getRandomInt() : Int {
      return rand_bert.nextInt(512) // todo: make less arbitrary?
    }

    fun formatTime(value: Double): String {
      return String.format("%.3f", value)
    }

    fun convertToBase64 (b: ByteArray): String {
      return android.util.Base64.encodeToString(b, android.util.Base64.NO_WRAP+android.util.Base64.URL_SAFE)
    }

    fun makeItText(): Boolean {
      return rand_do_text.nextFloat() < TEXT_PROBABILITY
    }
  }

}