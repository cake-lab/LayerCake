package edu.wpi.ssogden.layercakelibrary

import java.lang.Exception

open class InferenceException : Exception() {}

class RemoteModelNotRespondingException(s: String) : InferenceException() {}