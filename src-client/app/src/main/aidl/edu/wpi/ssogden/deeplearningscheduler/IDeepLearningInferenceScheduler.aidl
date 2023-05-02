// IDeepLearningInferenceScheduler.aidl
package edu.wpi.ssogden.deeplearningscheduler;

// Declare any non-default types here with import statements

interface IDeepLearningInferenceScheduler {
    /**
     * Demonstrates some basic types that you can use as parameters
     * and return values in AIDL.
     */
    //void basicTypes(int anInt, long aLong, boolean aBoolean, float aFloat,
    //        double aDouble, String aString);
    int getPid();

    String requestInference(String id, String dag_name);
    String requestInferencePriority(String id, String dag_name, int priority);
}