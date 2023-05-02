package edu.wpi.ssogden.layercakelibrary

import android.util.Log
import java.lang.Exception
import kotlin.concurrent.thread
import kotlin.math.roundToLong
import kotlin.system.measureNanoTime
import kotlin.system.measureTimeMillis

class Model(
  val application: Common.Companion.Application,
) {
  private val TAG: String = "Model"
  val modelVariants : MutableList<ModelVariant> = mutableListOf()
  val localModelVariants : MutableList<LocalModelVariant> = mutableListOf()

  var mostRecentRemoteVariants: List<SageMakerModelVariant> = listOf<SageMakerModelVariant>()
  var remoteVariantTimestamp = 0.0 // unix timestamp

  private fun updateRemoteModelVariantCache(minAccuracy: Double, maxLatency: Double) {
    mostRecentRemoteVariants = SageMakerModelVariant.getPossibleEndpoints(0.0, 10.0, application)
    remoteVariantTimestamp = Common.unixTime()
  }

  private fun getFastestLocalModelVariant(): LocalModelVariant {
    return localModelVariants.minByOrNull { it.latency }!!
  }

  private fun getRemoteModelVariants(minAccuracy: Double, maxLatency: Double, forceUpdate: Boolean = false): List<SageMakerModelVariant> {
    // todo: don't make this autoupdate if we piggyback requests
    //todo: check to see if we have enough time to update or if our maxLatency is too low

    val remoteVariantExpired = remoteVariantTimestamp + 60 < Common.unixTime()
    if (forceUpdate || remoteVariantExpired) {
      thread{
        updateRemoteModelVariantCache(minAccuracy, maxLatency)
      }.apply {
        if (forceUpdate) {
          this.join()
        }
      }
    }
    return mostRecentRemoteVariants
  }

  fun pickModelVariant(minAccuracy : Double, maxLatency: Double, request: InferenceRequest) : ModelVariant {

    // Special cases where we have no good local models, or are using a very small subset
    var potentialLocalVariants = localModelVariants
    if (potentialLocalVariants.none { it.latency <= maxLatency } && ! Common.ONLY_USE_REMOTE) {
      // If the latency is so low that no local models work then just use the fastest local model
      Log.w(TAG, "SLO too low: no local models available")
      return getFastestLocalModelVariant()
    }

    // Check for special case

    var bestLocalVariant = potentialLocalVariants
      .filter{ it.latency <= maxLatency && it.accuracy >= minAccuracy }
      .maxByOrNull { it.accuracy }

    if (bestLocalVariant == null) {
      // We don't just return here because we might just not have an accurate enough model locally
      bestLocalVariant = getFastestLocalModelVariant()
    }

    // Check for special case
    if (Common.ONLY_USE_LOCAL) {
      return bestLocalVariant
    }

    val estimatedNetworkTime = RemoteWrangler.estimateNetworkTime(request.dataSize)
    request.networkLatencyEstimate = estimatedNetworkTime
    Log.d(TAG, "estimatedNetworkTime: ${estimatedNetworkTime}")
    Log.d(TAG, "estimatedBandwidth: ${RemoteWrangler.estimatedBandwidth}")


    if (Common.ONLY_USE_REMOTE) {

      var remoteVariants : List<SageMakerModelVariant>? = null
      val timeToLoad = measureNanoTime {
        remoteVariants =
          getRemoteModelVariants(0.0, maxLatency - estimatedNetworkTime)!!
      } / Common.BILLION

      val timeBudget = Common.LATENCY_BUFFER * (maxLatency - timeToLoad - Common.RESIZE_BUFFER)

      // Filter out models that that don't work
      val possibleVariants = remoteVariants!!.filter {
        (it.accuracy >= minAccuracy) && it.isAvailable() &&
            (!it.isCold() && it.latency <= (timeBudget - estimatedNetworkTime))
      }
      Log.d(TAG, "mostRecentRemoteVariants: ${mostRecentRemoteVariants}")
      // Pick the best available, or return none if there are none
      val bestVariant = possibleVariants.maxByOrNull { it.accuracy }
        ?: mostRecentRemoteVariants.minByOrNull { it.latency }!!

      return bestVariant
    }

    var remoteVariants : List<SageMakerModelVariant>? = null
    val timeToLoad = measureNanoTime {
      if (Common.USE_BASELINE_MODEL) {
        remoteVariants =
          getRemoteModelVariants(bestLocalVariant.accuracy, maxLatency - estimatedNetworkTime)!!
      } else {

        remoteVariants =
          getRemoteModelVariants(0.0, maxLatency - estimatedNetworkTime)!!
      }
    } / Common.BILLION
    if (remoteVariants!!.isEmpty()) {
      // If there are no better remote models, immediately return the best local variant
      Log.w(TAG, "No remote models are better than local model")
      return bestLocalVariant
    }

    val allModelVariants : List<ModelVariant> =
      potentialLocalVariants + remoteVariants!! // include all local variants in case
    val timeBudget = Common.LATENCY_BUFFER * (maxLatency - timeToLoad - Common.RESIZE_BUFFER)

    // Filter out models that that don't work
    val possibleVariants = allModelVariants.filter {
      (it.accuracy >= minAccuracy) && it.isAvailable() && (
        ((it is SageMakerModelVariant) && (!it.isCold() && it.latency <= (timeBudget - estimatedNetworkTime)))
        ||
        ((it is LocalModelVariant) && it.hasFreeCapacity() && it.latency <= timeBudget)
      )
    }
    // Pick the best available, or return none if there are none
    val bestVariant = possibleVariants.maxByOrNull { it.accuracy }
      ?: return getFastestLocalModelVariant()

    return bestVariant
  }

  fun getAllRemoteVariants(forceUpdate: Boolean=true): List<SageMakerModelVariant> {
    return getRemoteModelVariants(0.0, 1000.0, forceUpdate)
  }

  fun execute(request: InferenceRequest, minAccuracy: Double, maxLatency: Double): ModelVariant? {

    var timeBudget = request.maxLatency - (Common.unixTime() - request.submissionTime!!)
    Log.d(
      TAG,
      "Adjusting time budget for ${request.id} from ${Common.formatTime(request.maxLatency)} (${maxLatency}) to ${timeBudget}"
    )
    var success: Boolean = false
    var modelVariant: ModelVariant? = null
    var startedBackup = false

    if (Common.USE_INFAAS) {
      // Uses proxy2
        try {
          val requestData = request.getRemoteData() as ImageData
          val networkTimeEstimate = RemoteWrangler.estimateNetworkTime(requestData.getSize())
          request.networkLatencyEstimate = networkTimeEstimate
          modelVariant = RemoteWrangler().runProxy2(requestData.getData(), (timeBudget-networkTimeEstimate), RemoteWrangler.SLOType.LATENCY)
          request.modelVariantsTried.add(modelVariant)
          request.modelVariantUsed = modelVariant
          completeExecution(request, modelVariant)
        } catch (e: Exception) {
          Log.e(TAG, e.stackTraceToString())
          modelVariant = getFastestLocalModelVariant()
          executeWithVariant(modelVariant, request)
        }

    } else {
      while ( ! success ) {
        timeBudget -= (measureNanoTime {
          Log.d(TAG, "Model selection took: ${
            measureTimeMillis {
              modelVariant = pickModelVariant(minAccuracy, timeBudget, request ) // todo: this will be slow as we don't initially have the image data
            }
          }")
          if ((Common.LAUNCH_BACKUP)
            || (Common.LAUNCH_LOCAL_BACKUP && modelVariant is SageMakerModelVariant)) {
              if (! startedBackup) {
                startedBackup = true
                startBackup(request, timeBudget)
              }
          }
          request.modelVariantsTried.add(modelVariant!!)
          Log.i(TAG, "Executing: ${request} with ${modelVariant}")
          success = executeWithVariant(modelVariant!!, request) //, request.getResizedJPEG(modelVariant!!.dimensions))
          Log.i(TAG, "Finished executing: ${request}")
        } / Common.BILLION)

        if ( ! success ) {
          Log.e(TAG, "ModelVariant (${modelVariant}) failed.  Retrying...")
        }
      }
    }
    return modelVariant
  }

  private fun executeWithVariant(modelVariant: ModelVariant, request: InferenceRequest, doUnload: Boolean =false): Boolean {
    if (request.isComplete()) {
      return true // Another process completed it beofre we even started
    }
    modelVariant.load()
    //var response: String? = null
    //Log.d(TAG, "Actually run in: ${ measureTimeMillis { response = modelVariant.execute(data) } }")
    val response = modelVariant.execute(request)

    if (response == null) {
      return false
    }
    completeExecution(request, modelVariant)
    if (doUnload) { modelVariant.unload() }
    return true
  }

  private fun completeExecution(request: InferenceRequest, modelVariant: ModelVariant) {

    try {
      request.editSemaphore.acquire()
      if (!request.isComplete()) {
        request.modelVariantUsed = modelVariant
        request.markComplete()
      }
    } finally {
      request.editSemaphore.release()
    }

  }

  private fun startBackup(request: InferenceRequest, timeBudget: Double) {
    val fastestVariant = getFastestLocalModelVariant()

    thread {
      var timeToSleep = 0L
      if (Common.DO_PROACTIVE_BACKUP) {
        timeToSleep = (1.1 * 1000 * (timeBudget - fastestVariant.latency)).roundToLong()
      } else {
        timeToSleep = (1000 * timeBudget).roundToLong()
      }
      if (timeToSleep > 0) {
        Thread.sleep(timeToSleep)
      }
      if (!request.isComplete()) {
        executeWithVariant(fastestVariant, request)
      }
    }
  }



}