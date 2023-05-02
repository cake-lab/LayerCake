package edu.wpi.ssogden.client

import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import androidx.annotation.RequiresApi
import edu.wpi.ssogden.layercakelibrary.*
import org.json.JSONObject
import java.util.concurrent.*
import kotlin.concurrent.thread
import kotlin.math.roundToLong
import kotlin.random.Random

class MainActivity : AppCompatActivity() {

  val TAG: String = "MainActivity"

  val scheduler: Scheduler = Scheduler(this)

  @RequiresApi(Build.VERSION_CODES.O)
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)

    var applicationsToTest: List<Common.Companion.Application>? = null
    if (Common.TEXT_PROBABILITY == 1.0) {
      applicationsToTest = listOf(Common.Companion.Application.TEXT)
    } else if (Common.TEXT_PROBABILITY == 0.0) {
      applicationsToTest = listOf(Common.Companion.Application.IMAGE)
    } else {
      applicationsToTest = listOf(Common.Companion.Application.IMAGE, Common.Companion.Application.TEXT)
    }

    if (Common.MEASURE_MODELS) {
      var a_thread : Thread? = null
      for (i in 0 until Common.NUM_MEASUREMENT_THREADS) {
        a_thread = thread {

          val a_scheduler: Scheduler = Scheduler(this)
          benchmarkTests(Common.NUM_MODEL_MEASUREMENTS, applicationsToTest, a_scheduler)
        }
      }
      a_thread!!.join()

    } else {
      thread { benchmarkTests(10, applicationsToTest) }.join()
      if (Common.RUN_SEQUENTIAL) {
        thread { runTestsSequential() }
      } else {
        thread { runTestsAsync() }
      }
    }
  }

  @RequiresApi(Build.VERSION_CODES.O)
  fun benchmarkTests(
    measurementsPerModel : Int,
    //applicationsToTest: List<Common.Companion.Application> = listOf(Common.Companion.Application.TEXT)
    applicationsToTest: List<Common.Companion.Application>,
    a_scheduler: Scheduler = scheduler
  ) {
    if (!Common.ONLY_USE_REMOTE) {
      a_scheduler.benchmarkLocalModels(
        measurementsPerModel,
        applicationsToTest = applicationsToTest
      )
    }

    if (!Common.ONLY_USE_LOCAL) {
      val wrangler: RemoteWrangler = RemoteWrangler()
      val threadPool = ScheduledThreadPoolExecutor(1)
      val networkUpdateTask = threadPool.scheduleAtFixedRate(
        {
          RemoteWrangler.refreshRTT()
          RemoteWrangler.refreshBandwidth()
        },
        0,
        10,
        TimeUnit.SECONDS
      )
      a_scheduler.benchmarkRemoteModels(measurementsPerModel, applicationsToTest=applicationsToTest)
      networkUpdateTask.cancel(false)
    }

    for (application in applicationsToTest) {
      Log.i(TAG, "Application: ${application}")
      val model = a_scheduler.modelByApplication[application]!!
      for (modelVariant: ModelVariant in model.localModelVariants.sortedBy { it.pathToModel }) {
        Log.i(
          TAG,
          "${modelVariant} : ${Common.formatTime(Common.mean(modelVariant.latencyMeasurements))} +/- ${
            Common.formatTime(Common.stddev(modelVariant.latencyMeasurements))
          }"
        )
      }
      for (modelVariant: ModelVariant in model.mostRecentRemoteVariants.sortedBy { it.modelName }) {
        Log.i(
          TAG,
          "${modelVariant} : ${Common.formatTime(Common.mean(modelVariant.latencyMeasurements))} +/- ${
            Common.formatTime(Common.stddev(modelVariant.latencyMeasurements))
          }"
        )
      }
    }
  }




  fun runTestsAsync() {
    val threadPool = ScheduledThreadPoolExecutor(4)
    val networkUpdateTask = threadPool.scheduleAtFixedRate(
      {
        RemoteWrangler.refreshRTT()
        RemoteWrangler.refreshBandwidth()
      },
      0,
      10,
      TimeUnit.SECONDS
    )

    val submittedRequests = LinkedBlockingQueue<InferenceRequest>() // mutableListOf<Common.InferenceRequest>()
    var nextOccurance = 15.0
    for (jobNum in 0 until (Common.NUM_BURNIN_REQUESTS + Common.NUM_JOBS_TO_RUN)) {
      if (jobNum == Common.NUM_BURNIN_REQUESTS) {
        nextOccurance += 15 // give it some time to let the requests finish of
      }
      nextOccurance += Common.getExpoNext(Common.ARRIVAL_RATE)
      //nextOccurance += 1 / Common.ARRIVAL_RATE
      threadPool.schedule(
        {

          var targetLatency : Double
          if (Common.DO_LATENCY_PIVOT) {
            if (200 > jobNum - Common.NUM_BURNIN_REQUESTS || jobNum - Common.NUM_BURNIN_REQUESTS > 800) {
              targetLatency = Common.getRandomDouble(Common.MIN_LATENCY, Common.MAX_LATENCY)
            } else if (400 > jobNum - Common.NUM_BURNIN_REQUESTS || jobNum - Common.NUM_BURNIN_REQUESTS > 600) {
              targetLatency =
                Common.getRandomDouble(Common.MIN_LATENCY, Common.Companion.PIVOT_LATENCY)
            } else {
              targetLatency = Common.getRandomDouble(Common.PIVOT_LATENCY, Common.MAX_LATENCY)
            }
          } else {
            targetLatency = Common.getRandomDouble()
          }

          var request: InferenceRequest
          if (Common.makeItText()) {
            request = TextRequest(
              this,
              Common.unixTime(),
              0.0,
              targetLatency
            )
          } else {
            request = ImageRequest(
              this,
              Common.unixTime(),
              0.0,
              targetLatency
            )
          }
          Log.i(TAG, "Created: ${request}")
          try {
            Log.i(TAG, "Added to queue: ${request} ${scheduler.addRequest(request)}")
          } catch (e: Exception) {
            Log.e(TAG, "Something went wrong adding ${request} to queue")
            Log.e(TAG, e.stackTraceToString())
          }
          if (jobNum >= Common.NUM_BURNIN_REQUESTS) {
            submittedRequests.add(request)
          }
        },
        (1000*nextOccurance).roundToLong(),
        TimeUnit.MILLISECONDS
      )
    }
    Log.d(TAG, "All jobs submitted")
    var jobsComplete = 0
    var numWithinSLO = 0
    var totalAccuracy = 0.0
    var numOnDevice = 0
    while (jobsComplete < Common.NUM_JOBS_TO_RUN) {
      val request = submittedRequests.take()
      request.complete.await()
      if (request.getResponseTime() <= request.maxLatency) {
        Log.i(
          TAG,
          "Completed #${++jobsComplete}: ${request.id} in ${Common.formatTime(request.getResponseTime())}s vs ${Common.formatTime(request.maxLatency)} (${
            Common.formatTime(request.getQueueTime())
          }s) (${Common.formatTime(request.getExecutionTime())}s) : ${request.modelVariantUsed}"
        )
        Log.i(TAG, "JobInfo: ${JSONObject(request.getCompleteInformation()).toString()}")
        numWithinSLO++
      } else {
        Log.e(
          TAG,
          "Completed #${++jobsComplete}: ${request.id} in ${Common.formatTime(request.getResponseTime())}s vs ${Common.formatTime(request.maxLatency)} (${
            Common.formatTime(request.getQueueTime())
          }s) (${Common.formatTime(request.getExecutionTime())}s) : ${request.modelVariantUsed}"
        )
        Log.e(TAG, "JobInfo: ${JSONObject(request.getCompleteInformation()).toString()}")
      }
      totalAccuracy += request.modelVariantUsed!!.accuracy
      if (request.modelVariantUsed is LocalModelVariant) {
        numOnDevice++
      }
    }
    networkUpdateTask.cancel(false)
    Log.i(TAG, "SLO Attainment: ${100 * numWithinSLO.toDouble() / Common.NUM_JOBS_TO_RUN}%")
    Log.i(TAG, "Effective Accuracy: ${100 * totalAccuracy / Common.NUM_JOBS_TO_RUN}%")
    Log.i(TAG, "Num On Device: ${100 * numOnDevice.toDouble() / Common.NUM_JOBS_TO_RUN}%")
    // todo: report accuracy, too
  }


  fun runTestsSequential() {
    val threadPool = ScheduledThreadPoolExecutor(4)
    val networkUpdateTask = threadPool.scheduleAtFixedRate(
      {
        RemoteWrangler.refreshRTT()
        RemoteWrangler.refreshBandwidth()
      },
      0,
      10,
      TimeUnit.SECONDS
    )
    var jobsComplete = 0
    var numWithinSLO = 0
    var totalAccuracy = 0.0
    var numOnDevice = 0
    for (jobNum in 0 until (Common.NUM_BURNIN_REQUESTS + Common.NUM_JOBS_TO_RUN)) {
      var request : InferenceRequest
      if (Common.makeItText()) {
        request = TextRequest(
          this,
          Common.unixTime(),
          0.0,
          Common.getRandomDouble()
        )
      } else {
        request = ImageRequest(
          this,
          Common.unixTime(),
          0.0,
          Common.getRandomDouble()
        )
      }
      Log.i(TAG, "Created: ${request}")
      try {
        Log.i(TAG, "Added to queue: ${request} ${scheduler.addRequest(request)}")
      } catch (e: Exception) {
        Log.e(TAG, "Something went wrong adding ${request} to queue")
        Log.e(TAG, e.stackTraceToString())
      }
      request.complete.await()

      if (jobNum >= Common.NUM_BURNIN_REQUESTS) {
        if (request.getResponseTime() <= request.maxLatency) {
          Log.i(
            TAG,
            "Completed #${++jobsComplete}: ${request.id} in ${Common.formatTime(request.getResponseTime())}s vs ${
              Common.formatTime(
                request.maxLatency
              )
            } (${
              Common.formatTime(request.getQueueTime())
            }s) (${Common.formatTime(request.getExecutionTime())}s) : ${request.modelVariantUsed}"
          )
          numWithinSLO++
        } else {
          Log.e(
            TAG,
            "Completed #${++jobsComplete}: ${request.id} in ${Common.formatTime(request.getResponseTime())}s vs ${
              Common.formatTime(
                request.maxLatency
              )
            } (${
              Common.formatTime(request.getQueueTime())
            }s) (${Common.formatTime(request.getExecutionTime())}s) : ${request.modelVariantUsed}"
          )
        }
        totalAccuracy += request.modelVariantUsed!!.accuracy
        if (request.modelVariantUsed is LocalModelVariant) {
          numOnDevice++
        }
      }
    }
    networkUpdateTask.cancel(false)
    Log.i(TAG, "SLO Attainment: ${100 * numWithinSLO.toDouble() / Common.NUM_JOBS_TO_RUN}%")
    Log.i(TAG, "Effective Accuracy: ${100 * totalAccuracy / Common.NUM_JOBS_TO_RUN}%")
    Log.i(TAG, "Num On Device: ${100 * numOnDevice.toDouble() / Common.NUM_JOBS_TO_RUN}%")
    // todo: report accuracy, too
  }

}