package opt.example;

import util.linalg.Vector;
import opt.EvaluationFunction;
import shared.Instance;

/**
 * A function that evaluates whether a vector represents a 2-colored graph
 * @author Daniel Cohen dcohen@gatech.edu
 * @version 1.0
 */
public class TwoColorsEvaluationFunction implements EvaluationFunction {


    private long functionCallCount;

    public TwoColorsEvaluationFunction() {
        this.functionCallCount = 0;
    }

    /**
     * @see opt.EvaluationFunction#getFunctionCallCount()
     */
    public long getFunctionCallCount()
    {
        return functionCallCount;
    }
    /**
     * @see opt.EvaluationFunction#setFunctionCallCount(opt.value)
     */
    public void setFunctionCallCount(long value)
    {
        functionCallCount = value;
    }

    /**
     * @see opt.EvaluationFunction#value(opt.OptimizationData)
     */
    public double value(Instance d) {
        functionCallCount++;
        Vector data = d.getData();
        double val = 0;
        for (int i = 1; i < data.size() - 1; i++) {
            if ((data.get(i) != data.get(i-1)) && (data.get(i) != data.get(i+1))) {
                val++;
            }
        }

        return val;
    }

}
