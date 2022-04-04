package ml.example;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.Evaluation;
import java.util.Random;
import java.io.File;

public class App 
{
    public static void main( String[] args )
    {
        try {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File("src/main/sources/diamonds.csv"));
            
            Instances data = loader.getDataSet();
            data.setClassIndex(6);


            int trainSize = (int) Math.round(data.numInstances() * 0.7);
            int testSize = data.numInstances() - trainSize;
                    
            Instances train_set = new Instances(data, 0, trainSize);
            Instances test_set = new Instances(data, trainSize, testSize);

            

            LinearRegression model = new LinearRegression();
            model.buildClassifier(train_set);

            Evaluation evaluation = new Evaluation(test_set);
            evaluation.evaluateModel(model,test_set);
            double rmse = evaluation.rootMeanSquaredError();

            System.out.println("RMSE: " + rmse);
        } catch (Exception ex) {
			System.err.println("Exception occurred! " + ex);
		}
}
}
