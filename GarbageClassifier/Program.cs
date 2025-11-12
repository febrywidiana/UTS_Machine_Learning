using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace GarbageClassifier
{
    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath { get; set; }

        [LoadColumn(1)]
        public string Label { get; set; }
    }

   public class ImagePrediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedLabel { get; set; } = string.Empty;

    [ColumnName("Score")]
    public float[] Score { get; set; } = Array.Empty<float>();
}


    class Program
    {
        static void Main(string[] args)
        {
            string datasetPath = @"C:\uts_ml\UTS_MACHINE_LEARNING\garbage_classification\"; // ganti sesuai folder kamu
            string modelPath = "garbageModel.zip";

            MLContext mlContext = new MLContext();

            Console.WriteLine("📂 Memuat gambar...");
            var images = mlContext.Data.LoadFromEnumerable(
                ImageLoader.LoadImagesFromDirectory(datasetPath, useFolderNameAsLabel: true));

            var split = mlContext.Data.TrainTestSplit(images, testFraction: 0.2);

            // Membuat folder workspace untuk cache ML.NET
string workspacePath = Path.Combine(Environment.CurrentDirectory, "mlnet_workspace");
Directory.CreateDirectory(workspacePath);


            Console.WriteLine("🧠 Melatih model...");
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.LoadRawImageBytes(
                    outputColumnName: "Image",
                    imageFolder: datasetPath,
                    inputColumnName: "ImagePath"))
                .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(
                    featureColumnName: "Image",
                    labelColumnName: "Label"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(split.TrainSet);

            Console.WriteLine("📊 Mengevaluasi model...");
            var predictions = model.Transform(split.TestSet);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine($"✅ MicroAccuracy: {metrics.MicroAccuracy:P2}");
            Console.WriteLine($"✅ MacroAccuracy: {metrics.MacroAccuracy:P2}");

            mlContext.Model.Save(model, split.TrainSet.Schema, modelPath);
            Console.WriteLine($"💾 Model tersimpan ke: {modelPath}");

            // Contoh prediksi gambar baru
            var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);

var sample = new ImageData()
{
    ImagePath = @"C:\uts_ml\UTS_MACHINE_LEARNING\garbage_classification\plastic\plastic1.jpg"
};

var prediction = predictor.Predict(sample);
Console.WriteLine($"🖼️ Gambar diprediksi sebagai: {prediction.PredictedLabel}");

        }
    }

    public static class ImageLoader
    {
        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
        {
            var files = Directory.GetFiles(folder, "*", searchOption: SearchOption.AllDirectories)
                .Where(f => Path.GetExtension(f) == ".jpg" || Path.GetExtension(f) == ".png" || Path.GetExtension(f) == ".jpeg");

            foreach (var file in files)
            {
                var label = useFolderNameAsLabel ? Directory.GetParent(file).Name : null;
                yield return new ImageData() { ImagePath = file, Label = label };
            }
        }
    }
}
