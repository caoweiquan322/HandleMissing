package com.fatty.ml;

import com.fatty.Helper;
import info.debatty.java.graphs.*;
import info.debatty.java.graphs.build.NNDescent;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

/**
 * Created by fatty on 16-8-24.
 */
public class FastLLR extends LLR {
    protected Graph<Instance> graph;
    protected int classIndex = -1;
    protected double speedup = 4.0;

    public FastLLR(int k, double speedup) {
        super(k);
        this.speedup = speedup;
    }

    public FastLLR(int k) {
        super(k);
    }

    public FastLLR() {
        super();
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        Helper.checkNotNull("instances", instances);
        if (instances.classIndex() < 0) {
            throw new Exception("The class index is not set yet.");
        }
        classIndex = instances.classIndex();

        completeData = new Instances(instances, 0);
        for (Instance instance : instances) {
            if (!instance.classIsMissing() && !instance.hasMissingValue()) {
                completeData.add(instance);
            }
        }
        if (completeData.numInstances() < MIN_INSTANCES_TO_TRAIN) {
            throw new Exception("Number of complete instances is too few to train a LLR classifier.");
        }

        // Build the k-NN graph.
        buildKNNGraph();
    }

    protected void buildKNNGraph() {
        // Create the nodes
        int count = completeData.numInstances();
        ArrayList<Node> nodes = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            // The value of our nodes will be weka instance.
            nodes.add(new Node<>(String.valueOf(i), completeData.get(i)));
        }

        // Instantiate and configure the build algorithm
        NNDescent builder = new NNDescent();
        builder.setK(initK);

        // early termination coefficient
        builder.setDelta(0.1);

        // sampling coefficient
        builder.setRho(0.2);

        builder.setMaxIterations(10);

        builder.setSimilarity(new SimilarityInterface<Instance>() {

            @Override
            public double similarity(Instance v1, Instance v2) {
                // Ignore instance compatibility checking.
                double diff = 0.0;
                double a;
                for (int i=0; i<v1.numAttributes(); ++i) {
                    if (i != classIndex && !v1.isMissing(i) && !v2.isMissing(i)) {
                        a = v1.value(i)-v2.value(i);
                        diff += a * a;
                    }
                }
                return 1.0 / (1.0 + Math.sqrt(diff));
            }
        });

        // Run the algorithm and get computed graph
        graph = builder.computeGraph(nodes);
    }

    /**
     * Note that the number k may to modified to adapt to needs.
     * @param instance
     * @param completeIndices
     * @param k
     * @return
     * @throws Exception
     */
    protected int[] calculateKNN(Instance instance, List<Integer> completeIndices, int k) throws Exception {
        // Calculate distances.
        NeighborList nl = graph.fastSearch(instance, k, speedup);

        if (nl.size() != k) { // Adapt k if necessary.
            if (nl.size() > 0)
                k = nl.size();
            else
                throw new Exception("Expected get " + k + " neighbors but got " + nl.size());
        }

        int[] indices = new int[k];
        int i=0;
        Neighbor nb;
        Iterator<Neighbor> itr = nl.iterator();
        while (itr.hasNext()) {
            nb = itr.next();
            indices[i] = Integer.parseInt(nb.node.id);
            ++i;
        }
        return indices;
    }

    protected double[] smoBasedLLRSolver(double[][] A, double b) {
        return null;
    }
}
