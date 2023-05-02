package edu.wpi.ssogden.layercakelibrary

import android.annotation.SuppressLint
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import com.amazonaws.auth.AWSCredentials
import com.amazonaws.auth.BasicAWSCredentials
import com.amazonaws.services.sagemakerruntime.AmazonSageMakerRuntimeAsyncClient
import com.amazonaws.services.sagemakerruntime.AmazonSageMakerRuntimeClient
import com.amazonaws.services.sagemakerruntime.model.InvokeEndpointRequest
import com.amazonaws.services.sagemakerruntime.model.InvokeEndpointResult
import com.beust.klaxon.Klaxon
import inference_service.InferenceGrpc
import inference_service.InferenceService
import io.grpc.ManagedChannel
import io.grpc.ManagedChannelBuilder
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.withTimeoutOrNull
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.lang.Exception
import java.nio.ByteBuffer
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread
import kotlin.random.Random
import kotlin.system.measureNanoTime
import kotlin.system.measureTimeMillis

class RemoteWrangler {


  enum class SLOType(val value : Int) {
    ACCURACY(1),
    LATENCY(2)
  }

  companion object {
    private val TAG: String = "RemoteWrangler"
    val concurrentUploads : Semaphore = Semaphore(Common.MAX_CONCURRENCY)

    // SageMakerProxy
    val channel: ManagedChannel = ManagedChannelBuilder.forAddress(Common.REMOTE_SERVER_IP,50051)
      .usePlaintext()
      .build()
    val stub: InferenceGrpc.InferenceBlockingStub = InferenceGrpc.newBlockingStub(channel)

    // SageMakerDirect
    val credentials : AWSCredentials = BasicAWSCredentials("AKIATWIRQCVEIIODPRNN", "FfxO7g9VQBSlYINJkEq6ERlb3mFGgys7FNvqePIM")
    //val sagemakerclient : AmazonSageMakerRuntimeClient = AmazonSageMakerRuntimeClient(credentials) // todo: credentials
    val sagemakerclient : AmazonSageMakerRuntimeAsyncClient = AmazonSageMakerRuntimeAsyncClient(credentials) // todo: credentials


    //todo: Set up caching for these and for models available, or update them
    var estimatedRTT = 0.0 //measureRTT()
    var rttTimestamp = 0.0
    var estimatedBandwidth = 0.001 //estimateBandwidth()
    var bandwidthTimestamp : Double = 0.0 //Common.unixTime()

    fun updateBandwidth(networkTime: Double, dataSizeInBytes: Int) {
      // todo: update the network latency
      Log.d(TAG, "Previous bandwidth: ${estimatedBandwidth / 1024}")
      if (networkTime - estimatedRTT > 0 ) {
        estimatedBandwidth =
          Common.EMA_ALPHA * estimatedBandwidth + (1 - Common.EMA_ALPHA) * dataSizeInBytes / (networkTime - estimatedRTT)
        bandwidthTimestamp = Common.unixTime()
      } else {
        Log.w(TAG, "Network time was calculated to be negative -- ignoring")
      }
      Log.d(TAG, "New bandwidth: ${estimatedBandwidth / 1024}")
    }

    @SuppressLint("CheckResult")
    fun measureProxyLatency(numBytes : Int): Double {
      val request = InferenceService.BandwidthMeasurementMessage.newBuilder()
        .setData(String(Random.nextBytes(numBytes)))
        .build()

      try {
        val roundTripTime = measureNanoTime {
          stub.bandwidthMeasurement(request)
        }
        return roundTripTime / Common.BILLION
      } catch (e: Exception) {
        Log.e(TAG, "Cannot connect to remote server.  Is it running?")
        return 0.01
      }
    }

    fun refreshRTT() {
      estimatedRTT = Common.EMA_ALPHA * estimatedRTT + (1-Common.EMA_ALPHA) * measureProxyLatency(0)
      rttTimestamp = Common.unixTime()
    }

    fun refreshBandwidth() {
      updateBandwidth(measureProxyLatency(Common.NUM_BYTES_FOR_BANDWIDTH_ESTIMATE), Common.NUM_BYTES_FOR_BANDWIDTH_ESTIMATE)
    }

    fun estimateNetworkTime(dataSizeInBytes: Int) : Double {

      if (rttTimestamp + Common.MODEL_CACHE_LIFETIME < Common.unixTime()) {
        thread {
          refreshRTT()
        }
      }
      return estimatedRTT + (dataSizeInBytes / estimatedBandwidth)
    }

  }

  fun runProxy1(modelName : String, inputData: ByteArray) : String {
    val request = InferenceService.Message1.newBuilder()
      .setApplication(InferenceService.Application.IMAGE)
      .setModelName(modelName.lowercase()) // todo: make sure this is standardized
      .setData(Common.convertToBase64(inputData))
      .build()

    var response: InferenceService.InferenceResponse? = null
    try {
      concurrentUploads.acquire()
      val networkTime = measureNanoTime { response = stub.infer1(request) } / Common.BILLION - response!!.metadata.processingLatency
      updateBandwidth(networkTime, request.serializedSize)
    } finally {
      concurrentUploads.release()
    }
    return response!!.getResponse(0)
  }

  fun runProxy2(inputData: ByteArray, sloTarget: Double, sloType: SLOType) : SageMakerModelVariant {
    val request = InferenceService.Message2.newBuilder()
      .setApplication(InferenceService.Application.IMAGE)
      .setData(Common.convertToBase64(inputData))
      .let {
        when (sloType) {
          SLOType.ACCURACY -> {it.setSloType(InferenceService.SLOType.ACCURACY)}
          SLOType.LATENCY -> {it.setSloType(InferenceService.SLOType.LATENCY)}
        }
      }.setSloValue(sloTarget)
      .build()

    var response: InferenceService.InferenceResponse? = null
    try {
      concurrentUploads.acquire()
      val networkTime = measureNanoTime { response = stub.infer2(request) } / Common.BILLION - response!!.metadata.processingLatency
      updateBandwidth(networkTime, request.serializedSize)
    } finally {
      concurrentUploads.release()
    }
    val endpointInfo = response!!.endpoint
    return SageMakerModelVariant(
      modelName = endpointInfo.modelName,
      endpointName = endpointInfo.endpointName,
      accuracy = endpointInfo.accuracy,
      latency = endpointInfo.latency,
      dimensions = endpointInfo.dimensions,
    )
    //return JSONObject(response!!.getResponse(0))["accuracy"] as String
  }

  fun runProxy3(minAccuracy: Double, maxLatency: Double, inputData: ByteArray? = null, application: Common.Companion.Application) : MutableList<SageMakerModelVariant> {
    val request = InferenceService.Message3.newBuilder()
      .setApplication(InferenceService.Application.IMAGE)
      .setAccuracySlo(minAccuracy)
      .setLatencySlo(maxLatency)
      .let {
        if (application == Common.Companion.Application.IMAGE) {
          it.setApplication(InferenceService.Application.IMAGE)
        } else {
          it.setApplication(InferenceService.Application.TEXT)
        }
      }
      .let {
        if (inputData != null) {
          it.setData(Common.convertToBase64(inputData)) // should only be used if fast connection is known or testing latency
        } else {
          it.setData("")
        }
      }.build()
    var response: InferenceService.Endpoints? = null
    try {
      concurrentUploads.acquire()
      val networkTime = measureNanoTime { response = stub.infer3(request) } / Common.BILLION - response!!.metadata.processingLatency
      updateBandwidth(networkTime, request.serializedSize)
    } catch (e: Exception) {
      Log.e(TAG, e.toString())
    } finally {
        concurrentUploads.release()
    }

    val endpoints : MutableList<SageMakerModelVariant> = mutableListOf<SageMakerModelVariant>()
    for (endpointInfo in response!!.endpointsList) {
      Log.d(TAG, "endpointInfo: ${endpointInfo}")
      if (endpointInfo.application == InferenceService.Application.IMAGE ) {
        if (application == Common.Companion.Application.IMAGE) {
          endpoints.add(
            SageMakerModelVariant(
              modelName = endpointInfo.modelName,
              endpointName = endpointInfo.endpointName,
              accuracy = endpointInfo.accuracy,
              latency = endpointInfo.latency,
              dimensions = endpointInfo.dimensions,
            )
          )
        }
      } else {
        if (application == Common.Companion.Application.TEXT) {
          endpoints.add(
            SageMakerModelVariantBert(
              modelName = endpointInfo.modelName,
              endpointName = endpointInfo.endpointName,
              accuracy = endpointInfo.accuracy,
              latency = endpointInfo.latency,
              dimensions = endpointInfo.dimensions,
            )
          )
        }
      }
    }
    return endpoints
  }

  @RequiresApi(Build.VERSION_CODES.O)
  fun runDirect(endpointName: String, jsonData : JSONObject) : String{

    //Log.i(TAG, "runDirect(${endpointName}, ${jsonData})")

    val sagemakerRequest = InvokeEndpointRequest()
      .withEndpointName(endpointName)
      .withContentType("application/json")
      .withBody(ByteBuffer.wrap(jsonData.toString().encodeToByteArray()))

    var response: InvokeEndpointResult? = null
    try {
      concurrentUploads.acquire()
      try {
        val responseFuture = sagemakerclient.invokeEndpointAsync(sagemakerRequest)
        try {
          response = responseFuture.get(Common.REMOTE_TIMEOUT, TimeUnit.SECONDS)
        } catch (e: InterruptedException) {
          Log.w(TAG, "Sagemaker timed out")
          responseFuture.cancel(true)
        }
        //response = sagemakerclient.invokeEndpoint(sagemakerRequest)
      } catch (e: Exception) {
        Log.w(TAG, "SageMakerDirect failed. (${e})")
      }
      if (response == null) {
        throw RemoteModelNotRespondingException("Remote model is failing to respond")
      }
    } finally {
      concurrentUploads.release()
    }
    try {
      var predictions =
        (JSONObject(String(response!!.body.array()))["predictions"] as JSONArray).getJSONArray(0)

      // Parse output
      var curr_max = predictions.getDouble(0)
      var curr_max_idx = 0
      for (i in 0 until predictions.length()) {
        val curr_val = predictions.getDouble(i)
        if (curr_val > curr_max) {
          curr_max = curr_val
          curr_max_idx = i
        }
      }
      return curr_max_idx.toString()
    } catch (e: JSONException) {
      return response.toString()
    }
  }



}