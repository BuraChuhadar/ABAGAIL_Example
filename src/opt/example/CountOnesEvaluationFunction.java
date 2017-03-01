package opt.example;

import util.linalg.Vector;
import opt.EvaluationFunction;
import shared.Instance;

/**
 * A function that counts the ones in the data
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class CountOnesEvaluationFunction implements EvaluationFunction {

    private long functionCallCount;

    public CountOnesEvaluationFunction() {
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
        for (int i = 0; i < data.size(); i++) {
            if (data.get(i) == 1) {
                val++;
            }
        }
        return val;
    }
}