using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace myApp
{
  class Program
  {
    public class SellData
    {
      [Column("0")] public float Age;

      [Column("1")] public float Hour;

      [Column("2")] [ColumnName("Label")] public string Label;
    }

    public class ProductPrediction
    {
      [ColumnName("PredictedLabel")] public string PredictedLabels;
    }


    public class IrisData
    {
      [Column("0")] public float SepalLength;


      [Column("1")] public float SepalWidth;

      [Column("2")] public float PetalLength;

      [Column("3")] public float PetalWidth;

      [Column("4")] [ColumnName("Label")] public string Label;
    }

    public class IrisPrediction
    {
      [ColumnName("PredictedLabel")] public string PredictedLabels;
    }

    static void Main(string[] args)
    {
      //Program.PredictIris();
      Program.PredictProduct();
    }

    public static void PredictProduct()
    {
      var pipeline = new LearningPipeline();
      string dataPath = "sell-data.txt";
      pipeline.Add(new TextLoader(dataPath).CreateFrom<SellData>(separator: ','));
      pipeline.Add(new Dictionarizer("Label"));
      pipeline.Add(new ColumnConcatenator("Features", "Age", "Hour"));
      pipeline.Add(new StochasticDualCoordinateAscentClassifier());
      pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });
      var model = pipeline.Train<SellData, ProductPrediction>();
      var prediction = model.Predict(new SellData()
      {
        Age = 12f,
        Hour = 1f
      });

      Console.WriteLine($"Predicted product is: {prediction.PredictedLabels}");
    }

    public static void PredictIris()
    {
      var pipeline = new LearningPipeline();
      string dataPath = "iris-data.txt";
      pipeline.Add(new TextLoader(dataPath).CreateFrom<IrisData>(separator: ','));
      pipeline.Add(new Dictionarizer("Label"));
      pipeline.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));
      pipeline.Add(new StochasticDualCoordinateAscentClassifier());
      pipeline.Add(new PredictedLabelColumnOriginalValueConverter() {PredictedLabelColumn = "PredictedLabel"});
      var model = pipeline.Train<IrisData, IrisPrediction>();
      var prediction = model.Predict(new IrisData()
      {
        SepalLength = 3.3f,
        SepalWidth = 1.6f,
        PetalLength = 0.2f,
        PetalWidth = 0.2f,
      });

      var prediction2 = model.Predict(new IrisData()
      {
        SepalLength = 5.8f,
        SepalWidth = 2.7f,
        PetalLength = 5.1f,
        PetalWidth = 1.9f
      });

      Console.WriteLine($"Predicred flower type is: {prediction.PredictedLabels}");

      Console.WriteLine($"Predicred 2 flower type is: {prediction2.PredictedLabels}");
    }
  }
}