package edu.wpi.ssogden.layercakelibrary

import android.content.Context
import android.util.Log
import java.util.concurrent.PriorityBlockingQueue
import kotlin.concurrent.thread
import kotlin.system.measureTimeMillis

class Scheduler (
  val context: Context,
  val numThreads: Int = 2,
) {
  private val TAG: String = "Scheduler"
  val modelByApplication = hashMapOf(
    Common.Companion.Application.IMAGE to Model(application = Common.Companion.Application.IMAGE).apply {
      this.localModelVariants.add(LocalModelVariant(context, "models/efficientnetb0.tflite", 0.71))
      this.localModelVariants.add(LocalModelVariant(context, "models/efficientnetb1.tflite", 0.791))
      this.localModelVariants.add(LocalModelVariant(context, "models/efficientnetb2.tflite", 0.801))
      this.localModelVariants.add(LocalModelVariant(context, "models/efficientnetb3.tflite", 0.816))
      this.localModelVariants.add(LocalModelVariant(context, "models/efficientnetb4.tflite", 0.829))
      this.localModelVariants.add(LocalModelVariant(context, "models/efficientnetb5.tflite", 0.836))
      this.localModelVariants.add(LocalModelVariant(context, "models/efficientnetb6.tflite", 0.840))
      this.localModelVariants.add(LocalModelVariant(context, "models/efficientnetb7.tflite", 0.843))
    },
  Common.Companion.Application.TEXT to Model(application = Common.Companion.Application.TEXT).apply {
    //this.localModelVariants.add(LocalModelVariantBert(context, "models/mobile_bert.tflite", 0.758))

//    // https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1
//    this.localModelVariants.add(LocalModelVariantBert(context, "models/lite-model_mobilebert_1_metadata_1.tflite", 0.7))
//
//    // https://tfhub.dev/tensorflow/lite-model/albert_lite_base/squadv1/metadata/1
//    this.localModelVariants.add(LocalModelVariantBert(context, "models/lite-model_albert_lite_base_squadv1_metadata_1.tflite", 0.65))

    // https://tfhub.dev/iree/lite-model/mobilebert/int8/1
    this.localModelVariants.add(LocalModelVariantBert(context, "models/lite-model_mobilebert_int8_1.tflite", 0.6))
    }
  )
  val queue : PriorityBlockingQueue<InferenceRequest> = PriorityBlockingQueue<InferenceRequest>()
  val threads : List<Thread> = listOf(
    thread { processQueue() },
    thread { processQueue() }
  )

  fun benchmarkLocalModels(
    numExecutions : Int = 10,
    applicationsToTest: List<Common.Companion.Application> = listOf(Common.Companion.Application.IMAGE, Common.Companion.Application.TEXT)
  ) {
    for (application in applicationsToTest) {
      val model = modelByApplication[application]!!
      for (modelVariant: ModelVariant in model.localModelVariants) {
        modelVariant.load()
        for (i in 0 until numExecutions) {
          val request = ImageRequest(
            context,
            Common.unixTime(),
            0.0,
            10.0
          )
          modelVariant.execute(request)
        }
        modelVariant.unload()
        Log.d(TAG, "${modelVariant}")
      }
    }
  }

  fun benchmarkRemoteModels(
    numExecutions : Int = 10,
    applicationsToTest: List<Common.Companion.Application> = listOf(Common.Companion.Application.IMAGE, Common.Companion.Application.TEXT)
  ) {
    for (application in applicationsToTest) {
      val model = modelByApplication[application]!!
      model.getAllRemoteVariants()
      for (modelVariant: ModelVariant in model.getAllRemoteVariants()) {
        modelVariant.load()
        for (i in 0 until numExecutions+1) {
          var request: InferenceRequest? = null
          if (application == Common.Companion.Application.IMAGE) {
            request = ImageRequest(
              context,
              Common.unixTime(),
              0.0,
              10.0
            )
          } else {
            request = TextRequest(
              context,
              Common.unixTime(),
              0.0,
              10.0
            )
          }
          modelVariant.execute(request)
          if (i == 0) {
            Thread.sleep(30000L) // to allow them to warm up after the firs tinference
          }
        }
        modelVariant.unload()
        Log.d(TAG, "${modelVariant}")
      }
    }
  }

  fun stepThroughLatency(minLatency: Int, maxLatency: Int, step: Int, repeatsAtEachStep: Int=1,
  modelsToTest: List<Common.Companion.Application> = listOf(Common.Companion.Application.IMAGE, Common.Companion.Application.TEXT)
  ) {
    for (application in modelsToTest) {
      val model = modelByApplication[application]!!
      for (latencyTarget in minLatency..maxLatency step step) {
        for (i in 0 until repeatsAtEachStep) {
          val request = ImageRequest(context, Common.unixTime(), 0.0, latencyTarget / 1000.0)
          val totalTime = measureTimeMillis {
            model.execute(request, 0.0, latencyTarget / 1000.0)
            request.complete.await()
          }
          if (totalTime <= latencyTarget) {
            Log.d(
              TAG,
              "Actual time: ${totalTime}ms (${latencyTarget}ms) (${request.modelVariantUsed})"
            )
          } else {
            Log.e(
              TAG,
              "Actual time: ${totalTime}ms (${latencyTarget}ms) (${request.modelVariantUsed})"
            )
          }
        }
      }
    }
  }

  private fun processQueue() {
    RemoteWrangler.refreshRTT()
    RemoteWrangler.refreshBandwidth()
    RemoteWrangler.refreshRTT()
    RemoteWrangler.refreshBandwidth()
    RemoteWrangler.refreshRTT()
    RemoteWrangler.refreshBandwidth()
    while (true) {
      val request : InferenceRequest = queue.take()
      Log.i(TAG, "Got from queue: ${request}")
      request.markExecution()
      var model: Model? = null
      if (request is ImageRequest) {
        model = modelByApplication[Common.Companion.Application.IMAGE]!!
      } else {
        model = modelByApplication[Common.Companion.Application.TEXT]!!
      }
      val variantUsed = model.execute(request, request.minAccuracy, request.maxLatency)
    }
  }

  fun addRequest(request: InferenceRequest): Boolean {
    Log.i(TAG, "Add to queue: ${request}")
    request.markSubmission()
    Log.d("CompareFunction", "${request} -> ${request.creationTime}")
    return queue.add(request)
  }

}