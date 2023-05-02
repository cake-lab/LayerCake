package edu.wpi.ssogden.utils

open class Resource(
  val name : String,
  val numThreads : Int,
  val useGPU : Boolean,
  val useRemote : Boolean
) : Comparable<Resource> {
  open val kind = KIND.GENERAL

  fun reserve() {}
  fun release() {}

  enum class KIND(val value: Int) {
    GENERAL(0),
    CPU(1),
    GPU(2),
    REMOTE(3);
  }

  override fun compareTo(other: Resource): Int {
    return (this.kind.value - other.kind.value)
  }
}

