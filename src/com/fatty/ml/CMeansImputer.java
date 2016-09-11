package com.fatty.ml;

import com.fatty.Helper;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.stream.Collectors;

/**
 * Created by fatty on 16/9/11.
 */
public class CMeansImputer extends AbstractImputer {
    @Override
    public Instances impute(Instances instances, int classIndex) throws ImputeException {
        Helper.checkNotNull("instances", instances);
        // Do imputing.
        try {
            Helper.setDataSetClassIndex(instances, classIndex);

            // K-means clustering.
            Instances complete = new Instances(instances, 0);
            complete.addAll(instances.stream().filter(inst -> !inst.hasMissingValue()).collect(Collectors.toList()));
            SimpleKMeans kMeans = new SimpleKMeans();
            kMeans.setOptions(Utils.splitOptions("-N 20")); // 20 centroids in total.
            complete.setClassIndex(-1);
            kMeans.buildClusterer(complete);
            Instances centroids = kMeans.getClusterCentroids();

            Instances imputed = new Instances(instances);
            LLR llr = new LLR(1);
            llr.setStrategy(LLR.LLRStrategy.Average);
            llr.buildClassifier(centroids);
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
