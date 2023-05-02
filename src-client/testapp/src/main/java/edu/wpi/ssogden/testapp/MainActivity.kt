package edu.wpi.ssogden.testapp

import android.content.ComponentName
import android.content.Intent
import android.content.ServiceConnection
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import android.view.View
import android.widget.Toast
import edu.wpi.ssogden.inferenceservice.IInferenceService
import edu.wpi.ssogden.inferenceservice.InferenceService
import edu.wpi.ssogden.inferenceservice.utils.PRIORITY
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicInteger
import kotlin.system.measureTimeMillis

class MainActivity : AppCompatActivity() {
  private val TAG: String = "MainActivity"

  var inferenceService : IInferenceService? = null
  var requestCounter : AtomicInteger = AtomicInteger(0)

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)
    initService()
  }

  override fun onDestroy() {
    super.onDestroy()
    stopService(Intent(this@MainActivity, InferenceService::class.java))
  }

  var mConnection = object : ServiceConnection {

    // Called when the connection with the service is established
    override fun onServiceConnected(className: ComponentName, service: IBinder) {
      // Following the example above for an AIDL interface,
      // this gets an instance of the IRemoteInterface, which we can use to call on the service
      inferenceService = IInferenceService.Stub.asInterface(service)
      Toast.makeText(this@MainActivity, "Service connected", Toast.LENGTH_LONG).show()

    }

    // Called when the connection with the service disconnects unexpectedly
    override fun onServiceDisconnected(className: ComponentName) {
      Log.e(TAG, "Service has unexpectedly disconnected")
      inferenceService = null
    }
  }

  fun onButtonClick(v : View) {
    Log.d(TAG, "Button clicked")
    GlobalScope.launch { executeInference() }
  }

  suspend fun executeInference() {
    val counter = requestCounter.getAndIncrement()
    var dag_priority = listOf(PRIORITY.LOW, PRIORITY.GENERAL, PRIORITY.HIGH)[counter%3]
    var dag_id = counter.toString()
    var dag_name = listOf("image_classification", "other_task")[0] //.random()
    dag_name = "simple"
    var result : String?
    var latency = measureTimeMillis {
      result = inferenceService?.requestInferencePriority(dag_id, dag_name, dag_priority.value)
    }
    withContext(Dispatchers.Main) {
      Toast.makeText(this@MainActivity, "$dag_id ($dag_priority): $dag_name: ${result} ${latency}ms ", Toast.LENGTH_SHORT).show()
      Log.d(TAG, "inference results: $dag_id ($dag_priority): $dag_name: ${result} ${latency}ms")
    }
  }

  private fun initService() {

    val i = Intent()
    i.setAction("edu.wpi.ssogden.inferenceservice.InferenceService")
    i.setClassName("edu.wpi.ssogden.inferenceserviceapp", "edu.wpi.ssogden.inferenceservice.InferenceService")

    //val i = Intent("edu.wpi.ssogden.inferenceservice.InferenceService.BIND")
    //i.setPackage("edu.wpi.ssogden.deeplearningscheduler")
    val ret = bindService(i, mConnection, BIND_AUTO_CREATE)
    Log.d(TAG, "initService() bound with $ret")
  }

  /** Unbinds this activity from the service.  */
  private fun releaseService() {
    unbindService(mConnection)
    Log.d(TAG, "releaseService() unbound.")
  }
}