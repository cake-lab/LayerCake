package edu.wpi.ssogden.layercakelibrary

import android.content.Context
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.recyclerview.widget.SortedList
import kotlinx.coroutines.*
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.lang.Exception
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.Semaphore
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.locks.Lock
import java.util.concurrent.locks.ReentrantLock
import kotlin.system.measureNanoTime
import kotlin.system.measureTimeMillis


abstract class ModelVariant (
    val name: String,
    val accuracy: Double,
    val dimensions: Int = Common.DEFAULT_MODEL_DIMENSIONS,
  ) {
  // todo: this class will hold the way to execute model variants
  // e.g. it will have the resources inside of it when loaded, or else will not
  // also, given a resource will estimate latency
  // each model variant will be a different ModelVariant object

  private val TAG: String = "ModelVariant"
  val latencyMeasurements: MutableList<Double> = mutableListOf()
  open var latency: Double = 0.0
  open val allowLatencyUpdate : Boolean = true

  fun updateLatency() {
    // todo: optimize so it doesn't always recalculate
    if (allowLatencyUpdate) {
      if (Common.PERCENTILE_OF_LATENCY > 0 && Common.PERCENTILE_OF_LATENCY < 1.0) {
        latency = Common.percentile(latencyMeasurements, Common.PERCENTILE_OF_LATENCY)
      } else {
        latency = Common.mean(latencyMeasurements)
      }
    }
  }

  val isLoaded : AtomicBoolean = AtomicBoolean(false)

  val loadLock : Lock = ReentrantLock()
  val availableInteractions : Semaphore = Semaphore(1)

  open fun hasFreeCapacity(): Boolean {
    return availableInteractions.availablePermits() > 0
  }

  fun load() {
    try {
      loadLock.lock()
      if (isLoaded.compareAndSet(false, true)) {
        _load()
      }
    } finally {
      loadLock.unlock()
    }
  }

  open fun execute(request: InferenceRequest): String? {
    var response : String? = null
    val data = request.getLocalData(dimensions) // we're just using trash data on-device since it doesn't matter
    try {
      availableInteractions.acquire()
      latencyMeasurements.add(measureNanoTime { response = _execute(data.getData()) } / Common.BILLION)
      updateLatency()
    } finally {
      availableInteractions.release()
    }
    return response
  }

  fun unload() {
    try {
      loadLock.lock()
      if (isLoaded.compareAndSet(true, false)) {
        _unload()
      }
    } finally {
      loadLock.unlock()
    }
  }

  fun isAvailable(): Boolean {
    return availableInteractions.availablePermits() > 0
  }

  abstract fun _load();
  abstract fun _execute(data: Any): String?;
  abstract fun _unload();

}

open class LocalModelVariant(
  val context: Context,
  val pathToModel: String,
  accuracy: Double,
  dimensions: Int = Common.DEFAULT_MODEL_DIMENSIONS,
  val numThreads: Int = 2
) : ModelVariant(pathToModel, accuracy, dimensions) {

  //private var interpreter : Interpreter? = null
  private val inputs: HashMap<Int, ByteBuffer> = hashMapOf()
  private val outputs: HashMap<Int, ByteBuffer> = hashMapOf()

  override fun toString(): String {
    return "LocalModelVariant<${pathToModel}, ${String.format("%.3f", latency)}s (${ if (latencyMeasurements.size > 0) {String.format("%.3f", latencyMeasurements.last())} else {
      "--"
    }}s)>"
  }

  override fun _load() {
    //interpreter = getInterpreter()
  }

  override fun _execute(data: Any): String? {
    val interpreter = loadInterpreter()
    addFakeIO(interpreter)
    interpreter.runForMultipleInputsOutputs(
      inputs.values.toTypedArray<ByteBuffer>(),
      outputs as Map<Int, Any>
    )
    GlobalScope.launch { interpreter.close() }
    return outputs[0].toString()
  }

  override fun _unload() {
    //interpreter!!.close()
    //interpreter = null
  }

  open protected fun loadInterpreter() : Interpreter {
    val modelBuffer : MappedByteBuffer = readModelBuffer()
    return loadCPUInterpreter(modelBuffer)
  }

  private fun readModelBuffer(): MappedByteBuffer {
    context.assets.openFd(pathToModel).use { fileDescriptor ->
      FileInputStream(fileDescriptor.fileDescriptor).use { inputStream ->
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(
          FileChannel.MapMode.READ_ONLY,
          startOffset,
          declaredLength
        )
      }
    }
  }

  private fun addFakeIO(interpreter: Interpreter) {
    // Set up fake buffers but never put data in them.  It'll run, but it'll be gibberish
    // This would be why BERT is running so fast -- it's java so they're 0 initilized so it immediately gets an EOL signal
    //Log.d(TAG, "interpreter: ${interpreter}")
    for (i in 0 until interpreter!!.inputTensorCount) {
      inputs[i] = ByteBuffer.allocate(interpreter!!.getInputTensor(i).numBytes())
    }
    for (i in 0 until interpreter!!.outputTensorCount) {
      outputs[i] = ByteBuffer.allocate(interpreter!!.getOutputTensor(i).numBytes())
    }
  }

  private fun loadCPUInterpreter(modelBuffer : MappedByteBuffer) : Interpreter {
    val options = Interpreter.Options()
    options.setNumThreads(numThreads)
    return Interpreter(modelBuffer, options)
  }

}


open class SageMakerModelVariant(
  val modelName : String,
  val endpointName : String,
  accuracy: Double,
  override var latency: Double, // todo: use 99th percentile
  dimensions: Int = Common.DEFAULT_MODEL_DIMENSIONS
) : ModelVariant(modelName, accuracy, dimensions) {

  // todo: record actual latency, less estimated network latency so we can run off that
  //  it will include error, which is good

  protected var lastColdHitTime: Double = 0.0
  override val allowLatencyUpdate: Boolean = true

  override fun toString(): String {
    return "SageMakerModelVariant<${modelName}, ${endpointName}, ${String.format("%.3f", latency)}s, (${isCold()})>"
  }
  companion object {
    internal val TAG: String = "SageMakerModelVariant"
    val remoteConnection : RemoteWrangler = RemoteWrangler()

    fun getPossibleEndpoints(minAccuracy: Double, maxLatency: Double, application: Common.Companion.Application) : MutableList<SageMakerModelVariant> {
      try {
        return remoteConnection.runProxy3(minAccuracy, maxLatency, application = application)
      } catch (e: Exception) {
        Log.e(TAG, "Failed to connect to Proxy.  Is it running?")
        return mutableListOf()
      }
    }
  }

  fun isCold(): Boolean {
    return lastColdHitTime + Common.TIME_TO_WARM_MODEL > Common.unixTime()
  }

  override fun _load() {
    // pass
  }

  override fun hasFreeCapacity(): Boolean {
    return true
  }

  @RequiresApi(Build.VERSION_CODES.O)
  override fun execute(request: InferenceRequest): String? {
    Log.d(TAG, "Executing: ${this}")
    var response : String? = null
    var data : RequestData
    if (Common.RESIZE_FOR_NETWORK) {
      data = request.getRemoteData(dimensions)
    } else {
      data = request.getRemoteData()
    }
    try {
      availableInteractions.acquire()
      val networkLatencyEstimate = RemoteWrangler.estimateNetworkTime(data.getSize())
      var remoteLatency : Double
      if (Common.MEASURE_MODELS) {
        remoteLatency =
          measureNanoTime { response = _execute(data.getData()) } / Common.BILLION // - networkLatencyEstimate
      } else {
        remoteLatency =
          measureNanoTime { response = _execute(data.getData()) } / Common.BILLION - networkLatencyEstimate
      }
      if (response != null ) {
        // Check to make sure it's a valid response and that the remote model didn't simply fail
        // If it did fail then don't add the time
        latencyMeasurements.add(remoteLatency)
        updateLatency()
      }
    } finally {
      availableInteractions.release()
    }
    return response
  }

  @RequiresApi(Build.VERSION_CODES.O)
  override fun _execute(data: Any) : String? {
    return try {
      var response: String? = null

      val jsonData = JSONObject(
        mapOf(
          Pair("instances",
            listOf(listOf(Common.convertToBase64(data as ByteArray)))
          )
        )
      )

      Log.d(TAG, "Actually run in: ${ 
        measureTimeMillis { 
          response = remoteConnection.runDirect(
            endpointName,
            jsonData
          ) 
        } 
      }")
      //remoteConnection.runDirect(endpointName, data)
      response
    } catch (e: RemoteModelNotRespondingException) {
      Log.e(TAG, "Model failed to respond")
      lastColdHitTime = Common.unixTime()
      null
    }
  }

  override fun _unload() {
    // pass
  }
}


class SageMakerModelVariantBert(
  modelName : String,
  endpointName : String,
  accuracy: Double,
  latency: Double,
  dimensions: Int = Common.DEFAULT_MODEL_DIMENSIONS
) : SageMakerModelVariant(modelName, endpointName, accuracy, latency, dimensions) {

  @RequiresApi(Build.VERSION_CODES.O)
  override fun _execute(data: Any) : String? {
    return try {
      var response: String? = null

      val jsonData = JSONObject(
        mapOf(
          Pair("inputs",
            JSONObject(data as HashMap<*, *>).toString()
          )
        )
      )

      Log.d(TAG, "Actually run in: ${
        measureTimeMillis {
          response = remoteConnection.runDirect(
            endpointName,
            jsonData
          )
        }
      }")
      //remoteConnection.runDirect(endpointName, data)
      response
    } catch (e: RemoteModelNotRespondingException) {
      Log.e(TAG, "Model failed to respond")
      lastColdHitTime = Common.unixTime()
      null
    }
  }


}
