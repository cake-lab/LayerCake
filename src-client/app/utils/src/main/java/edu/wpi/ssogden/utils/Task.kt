package edu.wpi.ssogden.utils


import java.nio.ByteBuffer
import java.util.logging.Logger


private val LOGGER = Logger.getLogger(Task::class.java.getName())

open class Task(
  val inputs: HashMap<Int, ByteBuffer>,
  val outputs: HashMap<Int, ByteBuffer>,
  val graph: ExecutionGraph,
  var priority: PRIORITY = PRIORITY.GENERAL
) : Runnable, Comparable<Task> {
  var running : Boolean = false
  var complete : Boolean = false
  var resource : Resource? = null

  var compareFunction : (Task, Task) -> Int = {
    t1 : Task, t2 : Task
      ->
    1 * (t1.priority.value - t2.priority.value)
  }

  // These assume that the underlaying ByteBuffers will be passed in and updated as needed
  //protected var inputs: Map<Int, ByteBuffer> = inputs
  //protected var outputs: Map<Int, ByteBuffer> = outputs

  // This should be updated done in a child so as to remove the dependency on tflite
  // The idea is that this runnable is everything that is needed to execute the operation, regardless of backing
  // It pulls in inputs and presents outputs
  open fun getRunnable(resource : Resource): Runnable { //(boolean useGPU, int numThreads) {}
    LOGGER.warning("Task not properly initilized")
    return Runnable {}
  }

  fun getInputBuffer(inputID: Int): ByteBuffer? {
    return inputs[inputID]
  }

  fun getOutputBuffer(outputID: Int): ByteBuffer? {
    return outputs[outputID]
  }

  override fun run() {
    TODO("Not yet implemented")
  }

  override fun compareTo(other: Task): Int {
    return compareFunction(this, other)
  }

  fun updateCompareFunction(newFunc: (Task, Task) -> Int) {
    compareFunction = newFunc
  }


}
