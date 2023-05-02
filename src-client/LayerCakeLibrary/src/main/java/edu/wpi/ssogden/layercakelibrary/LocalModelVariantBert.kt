package edu.wpi.ssogden.layercakelibrary


import android.content.Context
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import com.google.common.base.Joiner
import com.google.common.base.Verify.verify
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.bertqa.ml.Feature
import org.tensorflow.lite.examples.bertqa.ml.FeatureConverter
//import org.tensorflow.lite.examples.bertqa.ml.FeatureConverter
import org.tensorflow.lite.examples.bertqa.ml.ModelHelper
import org.tensorflow.lite.examples.bertqa.ml.QaAnswer
import org.tensorflow.lite.support.metadata.MetadataExtractor
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.util.*
import java.util.Collections.sort


private val SPACE_JOINER: Joiner = Joiner.on(" ")
//private Interpreter tflite;

class LocalModelVariantBert(
  context: Context,
  pathToModel: String,
  accuracy: Double,
  dimensions: Int = Common.DEFAULT_MODEL_DIMENSIONS,
  numThreads: Int = 2
) : LocalModelVariant(context, pathToModel, accuracy, dimensions, numThreads
) {

  val TAG = "LocalModelVariantBert"
  // Model from https://www.tensorflow.org/lite/examples/bert_qa/overview (but dug through code)

  // todo: set up https://github.com/huggingface/tflite-android-transformers/tree/master/bert

  companion object {

    private const val MAX_ANS_LEN = 32
    private const val MAX_QUERY_LEN = 64
    private const val MAX_SEQ_LEN = 384
    private const val DO_LOWER_CASE = true
    private const val PREDICT_ANS_NUM = 5

    private const val IDS_TENSOR_NAME = "ids"
    private const val MASK_TENSOR_NAME = "mask"
    private const val SEGMENT_IDS_TENSOR_NAME = "segment_ids"
    private const val END_LOGITS_TENSOR_NAME = "end_logits"
    private const val START_LOGITS_TENSOR_NAME = "start_logits"

    private const val OUTPUT_OFFSET = 1
  }

  private lateinit var featureConverter: FeatureConverter
  var dic: MutableMap<String, Int> = HashMap()
  private var interpreter : Interpreter? = null

  override fun _load() {
    //interpreter = getInterpreter()
  }

  @RequiresApi(Build.VERSION_CODES.N)
  override fun loadInterpreter(): Interpreter {
    val buffer: ByteBuffer = ModelHelper.loadModelFile(context, pathToModel)
    //val metadataExtractor = MetadataExtractor(buffer)
    //val loadedDic = ModelHelper.extractDictionary(metadataExtractor)
    //verify(loadedDic != null, "dic can't be null.")
    //dic.putAll(loadedDic)
    dic = loadDictionary()
    featureConverter = FeatureConverter(dic, DO_LOWER_CASE, MAX_QUERY_LEN, MAX_SEQ_LEN)
    val opt = Interpreter.Options()
    opt.setNumThreads(numThreads)
    return Interpreter(buffer, opt)
  }

  @RequiresApi(Build.VERSION_CODES.N)
  fun loadDictionary(): MutableMap<String, Int> {
    val loadedDic : MutableMap<String, Int> = mutableMapOf()
    context.assets.open("extras/vocab.txt").bufferedReader().apply {
      var i = 0
      for (line in this.lines()) {
        loadedDic[line.trim()] = i++;
      }
    }
    return loadedDic
  }

  override fun _execute(data: Any): String? {
    // todo: actually do
    return executeBERT("query string goes here", "content string goes here")
  }

  fun executeBERT(queryStr: String, contextStr: String): String? {
    val interpreter = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
      loadInterpreter()
    } else {
      TODO("VERSION.SDK_INT < N")
    }

    val inputIds = Array(1) {
      IntArray(
        MAX_SEQ_LEN
      )
    }
    val inputMask = Array(1) {
      IntArray(
        MAX_SEQ_LEN
      )
    }
    val segmentIds = Array(1) {
      IntArray(
        MAX_SEQ_LEN
      )
    }

    val startLogits = Array(1) {
      FloatArray(
        MAX_SEQ_LEN
      )
    }
    val endLogits = Array(1) {
      FloatArray(
        MAX_SEQ_LEN
      )
    }

    val feature: Feature = featureConverter.convert(queryStr, contextStr)

    for (j in 0 until MAX_SEQ_LEN) {
//      inputIds[0][j] = Common.getRandomInt()
//      inputMask[0][j] = Common.getRandomInt()
//      segmentIds[0][j] = Common.getRandomInt()
      inputIds[0][j] = feature.inputIds.get(j)
      inputMask[0][j] = feature.inputMask.get(j)
      segmentIds[0][j] = feature.segmentIds.get(j)
    }


    val inputs = arrayOf<Any>(
      inputIds,
      inputMask,
      segmentIds
    )

    for (i in 0 until MAX_SEQ_LEN) {
      if (inputIds[0][i] == 0) {
        continue
      }
    }

    val outputs = hashMapOf<Int, Any>(
      0 to startLogits,
      1 to endLogits
    )
    interpreter.runForMultipleInputsOutputs(inputs, outputs)

    //val answers: List<QaAnswer> = getBestAnswers(
    //  startLogits[0],
    //  endLogits[0], feature
    //)
    return endLogits[0].toString()
  }

  fun execute(interpreter: Interpreter, inputs: Array<Any>, outputs: HashMap<Int, Any>) {
    Log.d(TAG, "execute: ")
    interpreter.runForMultipleInputsOutputs(inputs, outputs)
  }

  @Synchronized
  private fun getBestAnswers(
    startLogits: FloatArray, endLogits: FloatArray, feature: Feature
  ): List<QaAnswer> {
    // Model uses the closed interval [start, end] for indices.
    val startIndexes = getBestIndex(startLogits)
    val endIndexes = getBestIndex(endLogits)
    val origResults: MutableList<QaAnswer.Pos> = ArrayList()
    for (start in startIndexes) {
      for (end in endIndexes) {
        if (!feature.tokenToOrigMap.containsKey(start + OUTPUT_OFFSET)) {
          continue
        }
        if (!feature.tokenToOrigMap.containsKey(end + OUTPUT_OFFSET)) {
          continue
        }
        if (end < start) {
          continue
        }
        val length = end - start + 1
        if (length > MAX_ANS_LEN) {
          continue
        }
        origResults.add(QaAnswer.Pos(start, end, startLogits[start] + endLogits[end]))
      }
    }
    origResults.sort()
    val answers: MutableList<QaAnswer> = ArrayList()
    for (i in origResults.indices) {
      if (i >= PREDICT_ANS_NUM) {
        break
      }
      var convertedText: String? = if (origResults[i].start > 0) {
        convertBack(feature, origResults[i].start, origResults[i].end)
      } else {
        ""
      }
      val ans = QaAnswer(convertedText, origResults[i])
      answers.add(ans)
    }
    return answers
  }

  /** Get the n-best logits from a list of all the logits.  */
  @Synchronized
  private fun getBestIndex(logits: FloatArray): IntArray {
    val tmpList: MutableList<QaAnswer.Pos> = ArrayList()
    for (i in 0 until MAX_SEQ_LEN) {
      tmpList.add(QaAnswer.Pos(i, i, logits[i]))
    }
    sort(tmpList)
    val indexes = IntArray(PREDICT_ANS_NUM)
    for (i in 0 until PREDICT_ANS_NUM) {
      indexes[i] = tmpList[i].start
    }
    return indexes
  }

  /** Convert the answer back to original text form.  */
  private fun convertBack(
    feature: Feature,
    start: Int,
    end: Int
  ): String? {
    // Shifted index is: index of logits + offset.
    val shiftedStart = start + OUTPUT_OFFSET
    val shiftedEnd = end + OUTPUT_OFFSET
    val startIndex = feature.tokenToOrigMap[shiftedStart]!!
    val endIndex = feature.tokenToOrigMap[shiftedEnd]!!
    // end + 1 for the closed interval.
    return SPACE_JOINER.join(feature.origTokens.subList(startIndex, endIndex + 1))
  }

}