// ======================================
// Project Name:weka_demo
// Package Name:com.kingyee.weka
// File Name:NaiveBayesTest.java
// Created Data:2016/8/30 16:15
// ======================================
package com.kingyee.weka;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;

/**
 * Created by cky on 2016/8/30 16:15.
 */
public class NaiveBayesTest {

    public static void main(String[] args) throws  Exception {

        File inputFile = new File(J48Test.class.getResource("/").getPath()+"/contact-lenses.arff");
        ArffLoader atf = new ArffLoader();
        atf.setFile(inputFile);
        Instances instancesTrain = atf.getDataSet();
        instancesTrain.setClassIndex(instancesTrain.numAttributes()-1);

        Classifier m_classifier = new NaiveBayes();
        m_classifier.buildClassifier(instancesTrain);

        Instance testInt;
        Evaluation testingEvalution = new Evaluation(instancesTrain);
        int length = instancesTrain.numInstances();

        for(int i = 0;i<length;i++){
            testInt = instancesTrain.instance(i);
            testingEvalution.evaluateModelOnceAndRecordPrediction(m_classifier,testInt);
        }

        System.out.println("分类器的正确率:"+ (1-testingEvalution.errorRate()));
    }
}
