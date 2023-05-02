package edu.wpi.ssogden.utils

import java.util.logging.Logger

private val LOGGER = Logger.getLogger(Task::class.java.getName())

class ExecutionGraph(
  var priority: PRIORITY = PRIORITY.GENERAL
    ) {
  var complete: Boolean = false
  var log = Logger.getGlobal()

  //val tasks : MutableList<Runnable> = mutableListOf()
  //val headTasks : MutableList<Task> = mutableListOf()
  val head : Task = Task(HashMap(), HashMap(), this).apply { this.complete = true }
  val children : MutableMap<Task, MutableList<Task>> = mutableMapOf(head to mutableListOf())

  fun getParentsMap(): MutableMap<Task, MutableList<Task>> {
    val parents : MutableMap<Task, MutableList<Task>> = mutableMapOf()
    for (child in children.values.flatten().toSet()) {
      parents[child] = children.filterKeys { children[it]!!.contains(child) }.keys.toMutableList()
    }
    return parents
  }

  fun addTask(task : Task, children : MutableList<Task> = mutableListOf(), isChild : Boolean = true) {
    this.children.put(task, children)
    if (!isChild) {
      this.children[head]!!.add(task)
    }
  }
  fun getChildren(t : Task) : List<Task>? {
    return this.children.get(t)
  }

  fun getReadyTasks(updatePriority : Boolean = false) : List<Task> {
    val parentsMap = getParentsMap()
    var readyTasks = parentsMap.filterKeys {
        child ->
          ( ! (child.running || child.complete) // Check if this task has already been started/completed
              &&
              parentsMap[child]!!.toList().all {
                parent -> parent.complete // Check of all of the parents of this task are completed
              })
    }.keys.toList()
    if (updatePriority) { readyTasks.map { t -> t.apply { priority = this@ExecutionGraph.priority } } }
    return readyTasks
  }

  // Comparator will come from scheduling function
  // Will likely include either a callback or a way to include the resources that are available
  // Utility is defined as in CremeBrulee, where higher is better (although may be reversed)
  //fun getNextTask(utility : Comparator<Task>, higherIsBetter : Boolean = true): Task? {
  //  return getTaskPriority(utility, higherIsBetter)[0]
  //}
  fun getTaskWithPriority(utility: Comparator<Task>, higherIsBetter : Boolean = true) : List<Task> {
    if (higherIsBetter) {
      return getReadyTasks().sortedWith(utility).reversed()
    } else {
      return getReadyTasks().sortedWith(utility)
    }
  }

  fun prepTask(t : Task) {
    // We want to grab the inputs from one model and put them in the next model here
    // Or if there is nothing, we can just pretend for now
    // But in the end, we'll have a document defining these connections
  }

  fun executeTask(task: Task, resource: Resource, callback: () -> Unit = {}): List<Task> {

    task.resource = resource
    task.running = true
    task.resource = resource
    task.run()
    task.running = true
    task.complete = true

    checkIfComplete()

    callback()

    return getChildren(task)!!.filter { it.complete }
  }

  private fun checkIfComplete() {
    val incompleteTasks = this.children.keys.filter { ! it.complete }
    LOGGER.info("Checking if complete: ${incompleteTasks.size}")
    if (incompleteTasks.size == 0) {
      this.complete = true
      LOGGER.info("Set status to complete")
    }
  }

}