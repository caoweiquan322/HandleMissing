package com.fatty.ml;

public class Main {

    public static void main(String[] args) {
        try {
            Misser misser = new KangMisser();
            misser.miss("hello", "world", 0.3);
        } catch (IllegalArgumentException | NullPointerException e) {
            System.out.println("Error occurs in main task. Details: " + e.toString());
        }
    }
}
