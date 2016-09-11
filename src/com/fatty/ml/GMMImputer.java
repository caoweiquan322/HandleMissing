package com.fatty.ml;

import com.fatty.Helper;
import weka.clusterers.EM;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.stream.Collectors;

/**
 * Created by fatty on 16/9/11.
 */
public class GMMImputer extends AbstractImputer {
    @Override
    public Instances impute(Instances instances, int classIndex) throws ImputeException {
        Helper.checkNotNull("instances", instances);
        // Do imputing.
        try {
            Helper.setDataSetClassIndex(instances, classIndex);

            // GMM clustering.
            Instances complete = new Instances(instances, 0);
            complete.addAll(instances.stream().filter(inst -> !inst.hasMissingValue()).collect(Collectors.toList()));

            EM em = new EM();
            em.setOptions(Utils.splitOptions("-N 5"));
            complete.setClassIndex(-1);
            em.buildClusterer(complete);
            double[][][] gmm = em.getClusterModelsNumericAtts();
            double[] prior = em.getClusterPriors();
            double[] avg = new double[gmm[0].length];
            for (int i=0; i<prior.length; ++i)
            {
                for (int j=0; j<gmm[0].length; ++j)
                    avg[j] += prior[i]*gmm[i][j][0];
            }
            for (int j=0; j<gmm[0].length; ++j)
                avg[j] /= prior.length;

            Instances imputed = new Instances(instances);
            for (Instance line: imputed) {
                if (line.hasMissingValue()) {
                    for (int j=0; j<line.numAttributes(); ++j) {
                        if (line.isMissing(j) && line.classIndex() != j) {
                            line.setValue(j, avg[j]);
                        }
                    }
                }
            }
            return imputed;
        } catch (Exception e) {
            throw new ImputeException("Error occurs while imputing data set. Details: " + e.getMessage(), e);
        }
    }
}
