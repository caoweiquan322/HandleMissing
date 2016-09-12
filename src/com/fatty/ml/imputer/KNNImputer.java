package com.fatty.ml.imputer;

import com.fatty.Helper;
import com.fatty.ml.UniformLLR;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by fatty on 16-8-31.
 */
public class KNNImputer extends AbstractImputer {
    @Override
    public Instances impute(Instances instances, int classIndex) throws ImputeException {
        Helper.checkNotNull("instances", instances);
        // Do imputing.
        try {
            Helper.setDataSetClassIndex(instances, classIndex);

            Instances imputed = new Instances(instances);
            UniformLLR llr = new UniformLLR(50, UniformLLR.NNStrategy.BruteForce, UniformLLR.LLRStrategy.Average); // K set to 20 by default.
            llr.buildClassifier(imputed);
            for (Instance line: imputed) {
                if (line.hasMissingValue()) {
                    llr.distributionForInstance(line);
                }
            }
            return imputed;
        } catch (Exception e) {
            throw new ImputeException("Error occurs while imputing data set. Details: " + e.getMessage(), e);
        }
    }
}
