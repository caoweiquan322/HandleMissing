package com.fatty.ml.imputer;

/**
 * Created by fatty on 16/8/21.
 */
public class ImputeException extends Exception {
    public ImputeException() {
        super();
    }

    public ImputeException(String message) {
        super(message);
    }

    public ImputeException(Throwable cause) {
        super(cause);
    }

    public ImputeException(String message, Throwable cause) {
        super(message, cause);
    }
}
