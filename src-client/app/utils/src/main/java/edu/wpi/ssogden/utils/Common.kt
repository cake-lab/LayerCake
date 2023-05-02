package edu.wpi.ssogden.utils

class Common {
}

enum class PRIORITY(val value: Int) {
  HIGH(0),
  GENERAL(1),
  LOW(2),;
  companion object {
    fun fromInt(value: Int) = PRIORITY.values().first { it.value == value }
  }
}