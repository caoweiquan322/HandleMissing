package com.fatty.ml;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import com.fatty.Helper;
import com.joptimizer.functions.ConvexMultivariateRealFunction;
import com.joptimizer.functions.LinearMultivariateRealFunction;
import info.debatty.java.graphs.*;
import info.debatty.java.graphs.build.NNDescent;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

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

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        Helper.checkNotNull("Trained complete data", completeData);
        Helper.checkPositive("Trained instances number", completeData.numInstances());
        if (!instance.equalHeaders(completeData.get(0))) {
            throw new Exception("The instance headers do not match.");
        }

        int numInstances = completeData.numInstances();
        int k = Math.min(initK, numInstances);
        int classIndex = completeData.classIndex();

        // Split attributes into complete/incomplete sets.
        List<Integer> completeIndices = new ArrayList<>();
        List<Integer> incompleteIndices = new ArrayList<>();
        for (int i=0; i<instance.numAttributes(); ++i) {
            if (instance.isMissing(i)) {
                incompleteIndices.add(i);
            } else if (instance.attribute(i).isNumeric() && i != classIndex) {
                completeIndices.add(i);
            }
        }
        if (completeIndices.size() < MIN_ATTRIBUTE_TO_RECONSTRUCT) {
            // Failed to classify.
            throw new Exception("The number of complete attributes is too few to reconstruct. At least "
                    + MIN_ATTRIBUTE_TO_RECONSTRUCT + " is required.");
        }

        // Get indices of k-NN.
        int[] nearestIndices = calculateKNN(instance, completeIndices, k);
        // In case there are not exactly k neighbors returned.
        if (k != nearestIndices.length)
            k = nearestIndices.length;

        // Solve the QP for the best reconstruction.
        double[] weights;
        if (strategy == LLRStrategy.Average) { // Average strategy.
            weights = new double[k];
            for (int i=0; i<k; ++i)
                weights[i] = 1.0/k;
        } else { // Optimization strategy.
            int d = completeIndices.size();
            double[][] halfP = new double[d+1][k];
            for (int j = 0; j < k; ++j) {
                Instance nb = completeData.get(nearestIndices[j]);
                for (int i = 0; i < d; ++i) {
                    halfP[i][j] = nb.value(completeIndices.get(i));
                }
                halfP[d][j] = 1.0;
            }
            double[] halfQ = new double[d+1];
            for (int i = 0; i < d; ++i) {
                halfQ[i] = -instance.value(completeIndices.get(i));
            }
            halfQ[d] = 1.0;

            //weights = emIterateForOptimalWeights(halfP, halfQ, nY, inequalities, A, B, k, completeIndices.size());
            weights = smoBasedLLRSolver(halfP, halfQ);
            Helper.checkNotNull("weights", weights);
            Helper.checkIntEqual(weights.length, k);
        }

        // Output the imputed data.
        double[] classHist = null;
        for (int i: incompleteIndices) {
            Attribute attr = instance.attribute(i);
            if (attr.isNumeric()) {
                double sum = 0.0;
                for (int j=0; j<k; ++j) {
                    sum += weights[j]*completeData.get(nearestIndices[j]).value(i);
                }
                if (instance.classIndex() == i) {
                    classHist = new double[] {sum};
                } else {
                    instance.setValue(i, sum);
                }
            } else if (attr.isNominal()) {
                Helper.checkPositive("attribute " + i + " values number", attr.numValues());
                double[] hist = new double[attr.numValues()];
                for (int j=0; j<k; ++j) {
                    hist[(int)Math.round(completeData.get(nearestIndices[j]).value(i))] += weights[j];
                }

                if (instance.classIndex() == i) {
                    classHist = hist;
                } else {
                    instance.setValue(i, Utils.maxIndex(hist));
                }
            } else {
                throw new Exception("We could only handle numeric/nominal attributes by now.");
            }
        }

        return classHist;
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

    public static double[] smoBasedLLRSolver(double[][] A, double[] b) {
        Helper.checkIntEqual(A.length, b.length);
        int k = A[0].length;
        double EPSILON = 1e-4;
        int MAX_ITR = 100;

        // Initialization.
        double[] w = new double[k];
        for (int i=0; i<k; ++i) w[i] = 1.0/k;
        double[][] At = matTranspose(A);
        double[][] AtA = matMultiply(At, A);
        double[] q = diagonal(AtA);
        double[] c = vecSub(matMultiply(A, w), b);
        double[] alpha = matMultiply(At, c);
        double[] cnst = null;// Need be initialized in each iteration.

        // Iterations.
        double sum0 = 0.0;
        for (double x: vecSub(matMultiply(A, w), b)) {
            //System.out.printf("%5.3f,", x);
            sum0 += Math.abs(x);
        }
        System.out.printf("Itr: %04d, %8.6f\n", 0, sum0);

        double delta = 0.0, nwi;
        for (int itr=0; itr<MAX_ITR; ++itr) {
            boolean hasBreak = false;
            for (int i = 0; i < k; ++i) {
                if (Math.abs(alpha[i]) > EPSILON && Math.abs(w[i]) > EPSILON) {
                    hasBreak = true;
                    cnst = vecSub(c, At[i], w[i], cnst); // Update the const part of the 1-d optimization problem.
                    nwi = -dotProd(At[i], cnst)/q[i]; // Update the solution vector.
                    if (nwi<0) nwi=0;
                    //if (nwi>1) nwi=1.0;
                    delta = nwi - w[i];
                    w[i] = nwi;
                    c = vecAdd(c, At[i], delta, c);
                    alpha = vecAdd(alpha, AtA[i], delta, alpha);
                }
            }

            // Display the partial results.
            double[] diff = vecSub(matMultiply(A, w), b);
            double sum = 0.0;
            for (double x: diff) {
                //System.out.printf("%5.3f,", x);
                sum += Math.abs(x);
            }
            System.out.printf("Itr: %04d, %8.6f\n", itr, sum);

            // Optimized.
            if (!hasBreak)
                break;
        }

        return w;
    }

    protected static double[] diagonal(double[][] A) {
        Helper.checkIntEqual(A.length, A[0].length);
        int n = A.length;
        double[] d = new double[n];
        for (int i=0; i<n; ++i)
            d[i] = A[i][i];
        return d;
    }

    protected static double[] vecSub(double[] a, double[] b) {
        return vecSub(a, b, 1.0, null);
    }

    protected static double[] vecSub(double[] a, double[] b, double[] c) {
        return vecSub(a, b, 1.0, c);
    }

    protected static double[] vecSub(double[] a, double[] b, double bScale, double[] c) {
        Helper.checkIntEqual(a.length, b.length);
        int n = a.length;
        if (c==null)
            c = new double[n];
        for (int i=0; i<n; ++i)
            c[i] = a[i]-b[i]*bScale;
        return c;
    }

    protected static double[] vecAdd(double[] a, double[] b) {
        return vecAdd(a, b, 1.0, null);
    }

    protected static double[] vecAdd(double[] a, double[] b, double[] c) {
        return vecAdd(a, b, 1.0, c);
    }

    protected static double[] vecAdd(double[] a, double[] b, double bScale, double[] c) {
        return vecSub(a, b, -bScale, c);
    }

    protected static double dotProd(double[] a, double[] b) {
        Helper.checkIntEqual(a.length, b.length);
        double prod = 0.0;
        for (int i=0; i<a.length; ++i)
            prod += a[i]*b[i];
        return prod;
    }
}
