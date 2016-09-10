package com.fatty.ml;

import info.debatty.java.graphs.*;
import info.debatty.java.graphs.build.NNDescent;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

/**
 * Created by fatty on 16-9-7.
 */
public class FastKNNDemo {
    public static void main(String[] args) {
        Random r = new Random();
        int count = 1000;
        int k = 10;

        // Create the nodes
        ArrayList<Node> nodes = new ArrayList<Node>(count);
        for (int i = 0; i < count; i++) {
            // The value of our nodes will be an int
            nodes.add(new Node<Integer>(String.valueOf(i), r.nextInt(10 * count)));
        }

        // Instantiate and configure the build algorithm
        NNDescent builder = new NNDescent();
        builder.setK(k);

        // early termination coefficient
        builder.setDelta(0.1);

        // sampling coefficient
        builder.setRho(0.2);

        builder.setMaxIterations(10);

        builder.setSimilarity(new SimilarityInterface<Integer>() {

            @Override
            public double similarity(Integer v1, Integer v2) {
                return 1.0 / (1.0 + Math.abs(v1 - v2));
            }
        });

        // Optionnallly, define a callback to get some feedback...
        builder.setCallback(new CallbackInterface() {

            @Override
            public void call(HashMap<String, Object> data) {
                System.out.println(data);
            }
        });

        // Run the algorithm and get computed graph
        Graph<Integer> graph = builder.computeGraph(nodes);

        // Display neighborlists
        for (Node n : nodes) {
            NeighborList nl = graph.get(n);
            System.out.print(n);
            System.out.println(nl);
        }

        // Optionnally, we can test the builder
        // This will compute the approximate graph, and then the exact graph
        // and compare results...
        builder.test(nodes);

        // Analyze the graph:
        // Count number of connected components
        System.out.println(graph.connectedComponents().size());

        // Search a query (fast approximative algorithm)
        System.out.println(graph.fastSearch(r.nextInt(10 * count), 1));

        // Count number of strongly connected components
        System.out.println(graph.stronglyConnectedComponents().size());

        // Now we can add a node to the graph (using a fast approximate algorithm)
        graph.fastAdd(
                new Node<Integer>("my new node 1", r.nextInt(10 * count)));
    }
}
