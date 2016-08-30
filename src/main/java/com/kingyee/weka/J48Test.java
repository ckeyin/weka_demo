// ======================================
// Project Name:weka_demo
// Package Name:com.kingyee.weka
// File Name:J48Test.java
// Created Data:2016/8/30 14:20
// ======================================
package com.kingyee.weka;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;

/**
 * Created by cky on 2016/8/30 14:20.
 */
public class J48Test {

    public static void main(String[] args) throws  Exception {

        Classifier m_classifier = new J48();
        /** 训练语料*/
        File inputFile = new File(J48Test.class.getResource("/").getPath()+"/cpu.with.vendor.arff");
        ArffLoader atf = new ArffLoader();
        atf.setFile(inputFile);
        Instances instancesTrain = atf.getDataSet();

        /** 读入训练文件*/
        inputFile = new File(J48Test.class.getResource("/").getPath()+"/cpu.with.vendor.arff");
        atf.setFile(inputFile);
        Instances instancesTest = atf.getDataSet();
        /** 设置分类属性所在行号 */
        instancesTest.setClassIndex(0);
        double num = instancesTest.numInstances();
        double right = 0.0f;
        instancesTrain.setClassIndex(0);
        /** 训练*/
        m_classifier.buildClassifier(instancesTrain);
        for(int i = 0;i <num;i++){
            /** 如果预测值和答案值相等*/
            if(m_classifier.classifyInstance(instancesTest.instance(i))
                    ==instancesTest.instance(i).classValue()){
                right++;
            }
        }

        System.out.println("J48 classification precision:"+ (right)/num);
    }
}
