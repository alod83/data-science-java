package ml.example;

import ml.comet.experiment.ExperimentBuilder;
import ml.comet.experiment.OnlineExperiment;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import java.util.Random;
import java.io.File;

public class App 
{
    public static void main( String[] args )
    {
        try {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File("src/main/source/breast-cancer.csv"));
            
            Instances data = loader.getDataSet();
            data.setClassIndex(1);


            int trainSize = (int) Math.round(data.numInstances() * 0.7);
            int testSize = data.numInstances() - trainSize;
                    
            Instances train_set = new Instances(data, 0, trainSize);
            Instances test_set = new Instances(data, trainSize, testSize);

            OnlineExperiment exp = ExperimentBuilder.OnlineExperiment()
                            .build();
            exp.setExperimentName("Decision Tree");

            Classifier cls = new J48();
            cls.buildClassifier(train_set);

            Evaluation eval = new Evaluation(test_set);
            eval.evaluateModel(cls,test_set);

            double precision = eval.precision(1);
            double recall = eval.recall(1);
            double accuracy = eval.pctCorrect();
                    
            exp.logMetric("precision", precision);
            exp.logMetric("recall", recall);
            exp.logMetric("accuracy", accuracy);

            exp.end();

        } catch (Exception ex) {
			System.err.println("Exception occurred! " + ex);
		}
    }
}
