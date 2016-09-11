package com.fatty.ml.misser;

/**
 * Created by fatty on 16/8/21.
 */
public class MissException extends Exception {
    public MissException() {
        super();
    }

    public MissException(String message) {
        super(message);
    }

    public MissException(Throwable cause) {
        super(cause);
    }

    public MissException(String message, Throwable cause) {
        super(message, cause);
    }
}
