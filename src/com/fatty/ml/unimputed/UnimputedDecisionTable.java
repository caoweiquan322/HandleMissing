package com.fatty.ml.unimputed;

import weka.classifiers.rules.DecisionTable;
import weka.core.Instances;

import java.util.stream.Collectors;

/**
 * Created by fatty on 16/9/10.
 */
public class UnimputedDecisionTable extends DecisionTable {
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        Instances pure = new Instances(instances, 0);
        pure.addAll(instances.stream().filter(instance -> !instance.hasMissingValue()).collect(Collectors.toList()));
        super.buildClassifier(pure);
    }
}
