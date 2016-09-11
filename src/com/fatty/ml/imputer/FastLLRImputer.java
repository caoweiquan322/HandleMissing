package com.fatty.ml.imputer;

import com.fatty.Helper;
import com.fatty.ml.FastLLR;
import com.fatty.ml.LLR;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by fatty on 16-9-7.
 */
public class FastLLRImputer extends AbstractImputer {
    @Override
    public Instances impute(Instances instances, int classIndex) throws ImputeException {
        Helper.checkNotNull("instances", instances);
        // Do imputing.
        try {
            Helper.setDataSetClassIndex(instances, classIndex);

            Instances imputed = new Instances(instances);
            LLR llr = new FastLLR(50); // K set to 20 by default.
            llr.setStrategy(LLR.LLRStrategy.Optimize);
            llr.buildClassifier(imputed);
            for (Instance line: imputed) {
                if (line.hasMissingValue()) {
                    llr.distributionForInstance(line);
                }
            }
            return imputed;
        } catch (Exception e) {
            e.printStackTrace();
            throw new ImputeException("Error occurs while imputing data set. Details: " + e.getMessage(), e);
        }
    }
}
