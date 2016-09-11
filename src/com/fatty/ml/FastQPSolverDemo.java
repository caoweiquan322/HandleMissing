package com.fatty.ml;

import java.util.Random;

/**
 * Created by fatty on 16/9/11.
 */
public class FastQPSolverDemo {
    public static void main(String[] args) {
        int k = 50;
        int d = 30;
        Random r = new Random(10);
        double[][] A = new double[d][k];
        double[] b = new double[d];
        for (int i=0; i<d; ++i) {
            for (int j=0; j<k; ++j) {
                A[i][j] = r.nextDouble();
            }
            b[i] = r.nextDouble();
        }
        FastLLR.smoBasedLLRSolver(A, b);

        // compare with slow solver.
        slowSolver(A, b);
    }

    public static void slowSolver(double[][] A, double[] b) {
        double[][] AtA = null;
    }
}
